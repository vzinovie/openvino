// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_deconv_node.h"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_input_node.h"
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include <ngraph/opsets/opset1.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <nodes/common/cpu_memcpy.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNDeconvolutionNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (std::dynamic_pointer_cast<const ngraph::opset1::ConvolutionBackpropData>(op) == nullptr &&
                std::dynamic_pointer_cast<const ngraph::opset1::GroupConvolutionBackpropData>(op) == nullptr) {
            errorMessage = "Only opset1 ConvolutionBackpropData and GroupConvolutionBackpropData operations are supported";
            return false;
        }
        size_t ndims = op->get_input_shape(0).size();
        if ((ndims < 3) || (ndims > 5)) {
            errorMessage = "Only 3D, 4D and 5D blobs are supported as input";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(const std::shared_ptr<ngraph::Node>& op,
                                                 const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_desc(0));
    });
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Deconvolution node with name '" + getName() + "'";

        auto convBackprop = std::dynamic_pointer_cast<const ngraph::opset1::ConvolutionBackpropData>(op);
        auto groupConvBackprop = std::dynamic_pointer_cast<const ngraph::opset1::GroupConvolutionBackpropData>(op);
        const auto dataShape = op->get_input_shape(0);
        weightDims = op->get_input_shape(1);
        const auto outShape = op->get_shape();
        OC = outShape[1];
        IC = dataShape[1];

        if (convBackprop) {
            algorithm = DeconvolutionCommon;

            groupNum = 1;
            withGroups = false;

            for (int i = 0; i < convBackprop->get_strides().size(); i++) {
                stride.push_back(static_cast<ptrdiff_t>(convBackprop->get_strides()[i]));
            }
            for (int i = 0; i < convBackprop->get_dilations().size(); i++) {
                dilation.push_back(static_cast<ptrdiff_t>(convBackprop->get_dilations()[i]) - 1);
            }
            paddingL = convBackprop->get_pads_begin();
            paddingR = convBackprop->get_pads_end();
        } else if (groupConvBackprop) {
            algorithm = DeconvolutionGrouped;

            groupNum = weightDims[0];
            withGroups = groupNum > 1;
            isDW = withGroups && groupNum == OC && groupNum == IC;

            for (int i = 0; i < groupConvBackprop->get_strides().size(); i++) {
                stride.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_strides()[i]));
            }
            for (int i = 0; i < groupConvBackprop->get_dilations().size(); i++) {
                dilation.push_back(static_cast<ptrdiff_t>(groupConvBackprop->get_dilations()[i]) - 1);
            }
            paddingL = groupConvBackprop->get_pads_begin();
            paddingR = groupConvBackprop->get_pads_end();
        }
        for (int i = 0; i < dilation.size(); i++) {
            kernel.push_back(weightDims[withGroups + 2 + i]);
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

InferenceEngine::Blob::Ptr MKLDNNDeconvolutionNode::createWeiBlobAsIO(InferenceEngine::SizeVector dims) {
    auto checkSize = [](size_t dst_size, size_t src_size) {
        if (dst_size < src_size) {
            IE_THROW() << "Cannot create internal buffer. Buffer can be overrun.";
        }
    };

    auto constNode = std::dynamic_pointer_cast<MKLDNNInputNode>(getParentEdgeAt(1)->getParent());
    if (!constNode)
        IE_THROW() << "Cannot cast const input node for node " << getName() << ".";
    auto blb = constNode->getConstBlob();
    if (!blb)
        IE_THROW() << "Cannot get const weights blob for node " << getName() << ".";

    // WA: In int8 case, we are processing weights using internal blob.
    // So we disconnect constant node containing weights from the graph and then don't use it.
    if (getParentEdges().size() == 3) {
        removeEdge(getParentEdgeAt(2));
        inDims.erase(inDims.begin() + 2);
    }
    removeEdge(getParentEdgeAt(1));
    inDims.erase(inDims.begin() + 1);

    InferenceEngine::SizeVector dimsForBlockedDesc{dims};
    std::swap(dimsForBlockedDesc[withGroups + 0], dimsForBlockedDesc[withGroups + 1]);

    InferenceEngine::SizeVector orderForBlockedDesc;
    if (withGroups) {
        orderForBlockedDesc = {0, 2, 1};
    } else {
        orderForBlockedDesc = {1, 0};
    }
    for (int i = 2 + withGroups; i < dimsForBlockedDesc.size(); i++)
        orderForBlockedDesc.push_back(i);

    BlockingDesc blkDesc(dimsForBlockedDesc, orderForBlockedDesc);
    InferenceEngine::TensorDesc tensorDesc(blb->getTensorDesc().getPrecision(), dims, blkDesc);

    auto fillInternalBlob = [&](char *data, size_t intBuffSize) {
        size_t offset = blb->byteSize();
        checkSize(intBuffSize, offset);
        cpu_memcpy_s(data, intBuffSize, blb->cbuffer(), blb->byteSize());
    };

    Blob::Ptr internalBlob = InferenceEngine::make_shared_blob<int8_t>(tensorDesc);
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    size_t intBuffSize = internalBlob->byteSize();

    fillInternalBlob(data, intBuffSize);

    return internalBlob;
}

bool MKLDNNDeconvolutionNode::canBeExecutedInInt8() {
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_common))
        return false;

    // todo: [antonvor] added these checks to fix performance problems
    if (kernel.size() == 3)
        return false;
    if (!withGroups && IC % 4 != 0 && OC % 4 != 0)
        return false;

    // todo: [antonvor] fusing is not supported yet for int8
    if (!fusedWith.empty())
        return false;

    for (int i = 0; i < kernel.size(); i++) {
        if (kernel[i] < stride[i])
            return false;
    }

    // not supported in oneDNN
    if (withGroups && !isDW && (IC % 16 != 0 || OC % 16 != 0))
        return false;

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inPrecision);

    InferenceEngine::Precision weiPrecision = getOriginalInputPrecisionAtPort(1);
    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(weiPrecision);

    if (isDW && (inputDataType == dnnl_s8 || dilation.size() == 3))
        return false;

    return (inputDataType == dnnl_s8 || inputDataType == dnnl_u8) && weightsDataType == dnnl_s8;
}

void MKLDNNDeconvolutionNode::getSupportedDescriptors() {
    if (!descs_fwd.empty() && !descs_bwd.empty())
        return;

    isInt8 = canBeExecutedInInt8();

    InferenceEngine::Precision inPrecision = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!isInt8) {
        if (!one_of(inPrecision, InferenceEngine::Precision::FP32, InferenceEngine::Precision::BF16))
            inPrecision = InferenceEngine::Precision::FP32;
        if (!one_of(outPrecision, InferenceEngine::Precision::FP32, InferenceEngine::Precision::BF16))
            outPrecision = InferenceEngine::Precision::FP32;
    }
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outPrecision);
    if (inputDataType == memory::data_type::bf16 || outputDataType == memory::data_type::bf16)
       inputDataType = outputDataType = memory::data_type::bf16;

    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";

    for (int i = 0; i < paddingR.size(); i++) {
        int with_group = getAlgorithm() == DeconvolutionGrouped ? 1 : 0;
        int krn = weightDims[with_group + 2 + i];
        int src = getChildEdgeAt(0)->getDims()[2 + i];
        int dst = getParentEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    if (isInt8) {
        //  WA: if int8 deconvolution is supported, we create internal weights blob in IO format
        std::swap(weightDims[withGroups + 0], weightDims[withGroups + 1]);
        internalBlobs.push_back(createWeiBlobAsIO(weightDims));
        auto format = getParentEdgeAt(0)->getDims().ndims() == 5 ? dnnl::memory::format_tag::ndhwc : dnnl::memory::format_tag::nhwc;
        MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, format);
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);
        createDescriptor({in_candidate}, {out_candidate});
    } else {
        for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
            MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, format);
            MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
    setPostOps(attr);
}

void MKLDNNDeconvolutionNode::setPostOps(mkldnn::primitive_attr &attr) {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops);
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void MKLDNNDeconvolutionNode::filterSupportedPrimitiveDescriptors() {
    MKLDNNNode::filterSupportedPrimitiveDescriptors();
    filterSupportedDescriptors();
}

void MKLDNNDeconvolutionNode::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            IE_THROW() << "Incorrect number of input or output memory formats for Deconvolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto src_tdesc = MKLDNNMemoryDesc(std::shared_ptr<dnnl::deconvolution_forward::desc>(*itd)->data.src_desc);
                    isSuitableDesc &= src_tdesc.isSame(inputMemoryFormatsFilter[0]);
                } else {
                    auto src_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_src_desc);
                    isSuitableDesc &= src_tdesc.isSame(inputMemoryFormatsFilter[0]);
                }
            }
            if (!outputMemoryFormatsFilter.empty()) {
                if (isInt8) {
                    auto dst_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::deconvolution_forward::desc>(*itd)->data.dst_desc);
                    isSuitableDesc &= dst_tdesc.isSame(outputMemoryFormatsFilter[0]);
                } else {
                    auto dst_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_dst_desc);
                    isSuitableDesc &= dst_tdesc.isSame(outputMemoryFormatsFilter[0]);
                }
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool MKLDNNDeconvolutionNode::created() const {
    return getType() == Deconvolution;
}

void MKLDNNDeconvolutionNode::createPrimitive() {
    if (prim)
        return;

    if (isInt8) {
        auto prim_desc = createPrimitiveDescriptor<deconvolution_forward::primitive_desc,
                deconvolution_forward::desc>(attr);

        prim.reset(new deconvolution_forward(prim_desc));

        auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, internalBlobMemory[0]->GetPrimitive()}, {DNNL_ARG_DST, dst}};
    } else {
        auto prim_desc = createPrimitiveDescriptor<convolution_backward_data::primitive_desc,
                convolution_backward_data::desc, convolution_forward::primitive_desc>(attr);

        prim.reset(new convolution_backward_data(prim_desc));

        auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        auto weights = getParentEdgeAt(1)->getMemory().GetPrimitive();
        auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        primArgs = {{DNNL_ARG_DIFF_DST, src}, {DNNL_ARG_WEIGHTS, weights}, {DNNL_ARG_DIFF_SRC, dst}};
    }
}

void MKLDNNDeconvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                               const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    // grouping and autoblicking is not compatible
    if ((withGroups && !isDW) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    auto convert = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };

    if (isInt8) {
        MKLDNNDims weightsDims = MKLDNNDims(weightDims);
        MKLDNNMemoryDesc wgh_candidate{weightsDims, memory::data_type::s8, memory::format_tag::any};
        std::shared_ptr<mkldnn::deconvolution_forward::desc> deconv_desc;
        deconv_desc.reset(new deconvolution_forward::desc(prop_kind::forward_inference, mkldnn::algorithm::deconvolution_direct,
                                                          in_candidate, wgh_candidate, out_candidate,
                                                          convert(stride), convert(dilation),
                                                          convert(paddingL), convert(paddingR)));
        descs.emplace_back(deconv_desc);
    } else {
        MKLDNNDims weightsDims = MKLDNNDims(weightDims);
        MKLDNNMemoryDesc wgh_candidate{weightsDims, in_candidate.getDataType(), memory::format_tag::any};
        for (auto alg : {mkldnn::algorithm::convolution_winograd, mkldnn::algorithm::convolution_direct}) {
            std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_inference, alg,
                                                          out_candidate, wgh_candidate, in_candidate,
                                                          convert(stride),
                                                          convert(dilation),
                                                          convert(paddingL),
                                                          convert(paddingR)));

            std::shared_ptr<mkldnn::convolution_backward_data::desc> deconv_desc;
            deconv_desc.reset(new convolution_backward_data::desc(alg, out_candidate, wgh_candidate,
                                                                  in_candidate,
                                                                  convert(stride),
                                                                  convert(dilation),
                                                                  convert(paddingL),
                                                                  convert(paddingR)));
            descs_fwd.push_back(conv_desc);
            descs_bwd.push_back(deconv_desc);

            auto fwd_conv_pd = std::make_shared<convolution_forward::primitive_desc>(*conv_desc, getEngine(), true);
            if (fwd_conv_pd->get(true) == nullptr)
                continue;

            descs.emplace_back(deconv_desc, fwd_conv_pd);
        }
    }
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    if (idx == 2) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(getOriginalInputPrecisionAtPort(2),
                                                            getParentEdgeAt(2)->getDims().ToSizeVector(),
                                                            TensorDesc::getLayoutByDims(getParentEdgeAt(2)->getDims().ToSizeVector())));
    }

    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
            : isInt8 ? MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx)) : MKLDNNMemoryDesc(primitive_desc_it.diff_dst_desc(idx));

    if (desc.getLayout() == InferenceEngine::Layout::ANY) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    } else {
        if (getParentEdgeAt(idx)->getDims().ToSizeVector().size() != *std::max_element(desc.getBlockingDesc().getOrder().begin(),
                                                                                       desc.getBlockingDesc().getOrder().end()) + 1) {
            auto old_dims = getParentEdgeAt(idx)->getDims().ToSizeVector();
            auto new_dims = weightDims;

            auto td = InferenceEngine::TensorDesc(desc.getPrecision(),
                                                  new_dims,
                                                  desc.getBlockingDesc());
            if (new_dims.size() == desc.getBlockingDesc().getBlockDims().size()) {
                td.setLayout(BLOCKED);
            }
            return MKLDNNMemoryDesc(td);
        } else {
            return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                                getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                                desc.getBlockingDesc()));
        }
    }
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = isInt8 ? MKLDNNMemoryDesc(primitive_desc_it.dst_desc(idx))
            : MKLDNNMemoryDesc(primitive_desc_it.diff_src_desc(idx));
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

InferenceEngine::Precision MKLDNNDeconvolutionNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return MKLDNNExtensionUtils::getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNDeconvolutionNode, Deconvolution);

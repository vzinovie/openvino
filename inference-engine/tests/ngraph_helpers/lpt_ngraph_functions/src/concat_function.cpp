// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/concat_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat->set_friendly_name("output");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat->set_friendly_name("output");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");
    const auto fakeQuantize3 = makeFakeQuantize(input3, precision, fqOnData3);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { fakeQuantize1->output(0), fakeQuantize2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat1");

    const auto concat2 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { fakeQuantize2->output(0), fakeQuantize3->output(0) },
        1ull);
    concat2->set_friendly_name("concat2");

    auto& rtInfo2 = concat2->get_rt_info();
    rtInfo2["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat2");

    const ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat1),
        std::make_shared<ngraph::opset1::Result>(concat2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector { input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    } else {
        auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        intermediateOp,
        weights,
        ngraph::Strides { 1, 1 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::Strides { 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithSplitedIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    size_t numSplit = 2;
    size_t splitedAxis = 1;


    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1] / numSplit,
        inputShape[2],
        inputShape[3]
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<ngraph::op::Op> intermediateOp;

    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    intermediateOp = std::make_shared<ngraph::opset1::Split>(fakeQuantize2->output(0), constant, numSplit);

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, splitedAxis);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1] / numSplit, inputShape[1] / numSplit, 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        intermediateOp->output(1),
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution),
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalSelectionWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    } else {
        auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        intermediateOp,
        weights,
        ngraph::Strides { 1, 1 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::Strides { 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithDifferentPrecisionOnChilds(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        concat->output(0),
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("AvgPool");

    const auto maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        concat->output(0),
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    maxPool->set_friendly_name("MaxPool");

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(avgPool));
    results.push_back(std::make_shared<ngraph::opset1::Result>(maxPool));

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithDifferentChildsTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithIntermediateWithConstant(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");


    std::shared_ptr<ngraph::op::Op> intermediateOp;

    if (transparentIntermediate) {
        const auto pooling = makeMaxPool(fakeQuantize1->output(0), { 3, 3 });

        ngraph::op::v0::InterpolateAttrs attributes;
        attributes.axes = ngraph::AxisSet{ 2, 3 };
        attributes.mode = "nearest";
        attributes.align_corners = false;
        attributes.antialias = false;
        attributes.pads_begin = { 0 };
        attributes.pads_end = { 0 };
        const auto outputShape = op::Constant::create(
            ngraph::element::i64, ngraph::Shape{ 2 },
            ngraph::Shape{ inputShape[2], inputShape[3] });
        intermediateOp = std::make_shared<ngraph::opset1::Interpolate>(pooling->output(0), outputShape, attributes);
        intermediateOp->set_friendly_name("intermediate");
    } else {
        intermediateOp = fakeQuantize1;
    }

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize2->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(concat),
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateWithConstantTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithReshapeAtTheEndTransformation(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const FakeQuantizeOnDataWithConstant& fqOnData3) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ngraph::opset1::Concat> concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat1->set_friendly_name("concat1");

    const std::shared_ptr<Node> intermediate = makeMaxPool(concat1->output(0), {1ul, 1ul});

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");
    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const std::shared_ptr<ngraph::opset1::Concat> concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ fakeQuantize3, intermediate }, 1);
    concat2->set_friendly_name("concat2");

    const Shape concat2Shape = concat2->output(0).get_shape();
    const std::shared_ptr<Node> maxPool = makeMaxPool(concat2->output(0), {concat2Shape[2], concat2Shape[3]});
    const std::shared_ptr<Node> reshape = std::make_shared<ngraph::opset1::Reshape>(
        maxPool,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2ul}, std::vector<size_t>{0, 0}),
        true);
    reshape->set_friendly_name("output");


    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape)};

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2, input3 },
        "OriginalWithReshapeAtTheEndTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations& dequantizationOperations) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Concat>>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization = makeDequantization(concat, dequantizationOperations);
    lastDequantization->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    if (fqOnData1.outputPrecision != fqOnData2.outputPrecision) {
        throw std::runtime_error("FakeQuantize expected precisions are different");
    }
    const ngraph::element::Type fqOnDataPrecision = fqOnData1.outputPrecision;
    if (fqOnDataPrecision != ngraph::element::undefined) {
        if (fakeQuantize1->get_output_element_type(0) != fakeQuantize2->get_output_element_type(0)) {
            throw std::runtime_error("FakeQuantize operation precisions are different");
        }
        const ngraph::element::Type fakeQuantizePrecision = fakeQuantize1->get_output_element_type(0);

        if (fqOnDataPrecision != fakeQuantizePrecision) {
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, fqOnDataPrecision);
        }
    }

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReference(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(input1, inputPrecision, fqOnData1);
    if (precisionBeforeOp.is_real()) {
        low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, inputPrecision);
    } else {
        low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    }
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(input2, inputPrecision, fqOnData2);
    if (precisionBeforeOp.is_real()) {
        low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, inputPrecision);
    } else {
        low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    }
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Concat>>(
        ngraph::OutputVector{ deqBefore1, deqBefore2 }, 1);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");
    if (precisionAfterOperation.is_real()) {
        low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, inputPrecision);
    } else {
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);
    }

    auto deqStructure = dequantizationAfter;
    deqStructure.multiply.outPrecision = inputPrecision;
    if (inputPrecision != element::f32) {
        deqStructure.convert = DequantizationOperations::Convert(element::f32);
    }

    const auto lastDequantization = makeDequantization(concat, deqStructure);
    lastDequantization->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input3->set_friendly_name("input3");

    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
    fakeQuantize3->set_friendly_name("fakeQuantize3");
    const auto deqBefore3 = makeDequantization(fakeQuantize3, dequantizationBefore);

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, deqBefore2 },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat1");

    const auto concat2 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore2, deqBefore3 },
        1ull);
    concat2->set_friendly_name("concat2");

    auto& rtInfo2 = concat2->get_rt_info();
    rtInfo2["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat2");

    const std::shared_ptr<ngraph::Node> lastDequantization1 = makeDequantization(concat1, dequantizationOperations1);
    lastDequantization1->set_friendly_name("concat1");

    const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(concat2, dequantizationOperations2);
    lastDequantization2->set_friendly_name("concat2");

    const ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(lastDequantization2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector { input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore1);

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(deqBefore2, { 3, 3 });
    } else {
        const auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            deqBefore2,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, intermediateOp },
        1);
    concat->set_friendly_name("concat");
    low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization1 = makeDequantization(concat, dequantizationAfter1);
    lastDequantization1->set_friendly_name("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(intermediateOp, dequantizationAfter2);

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        lastDequantization2,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithSplitedIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    size_t numSplit = 2;
    size_t splitedAxis = 1;

    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1] / numSplit,
        inputShape[2],
        inputShape[3]
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionAfterOperation);
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);


    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionAfterOperation);
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore1);

    std::shared_ptr<ngraph::op::Op> intermediateOp;

    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    intermediateOp = std::make_shared<ngraph::opset1::Split>(deqBefore2, constant, numSplit);

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ deqBefore1, intermediateOp->output(0) }, splitedAxis);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto lastDequantization1 = makeDequantization(concat, dequantizationOperations1);
    const auto lastDequantization2 = makeDequantization(intermediateOp->output(1), dequantizationOperations2);

    auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ inputShape[1] / numSplit, inputShape[1] / numSplit, 1, 1 }, { 1 });

    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        lastDequantization2,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceSelectionWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore2);

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(deqBefore2, { 3, 3 });
    } else {
        auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, intermediateOp->output(0) },
        1);
    concat->set_friendly_name("concat");
    low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization1 = dequantizationOperations1.empty() ?
        concat :
        makeDequantization(concat, dequantizationOperations1);
    lastDequantization1->set_friendly_name("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization2 = dequantizationOperations2.empty() ?
        nullptr :
        makeDequantization(intermediateOp, dequantizationOperations2);

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        lastDequantization2 == nullptr ? intermediateOp : lastDequantization2,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithDifferentPrecisionOnChilds(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool multiChannel,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ deqBefore1, deqBefore2 }, 1);
    low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto lastDequantization1 = makeDequantization(concat->output(0), dequantizationAfter1);

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        lastDequantization1,
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("AvgPool");

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(avgPool));

    if (!dequantizationAfter2.empty()) {
        const std::shared_ptr<ngraph::opset1::MaxPool> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
            concat->output(0),
            stride,
            padBegin,
            padEnd,
            kernel,
            roundingType,
            padType);

        const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(maxPool, dequantizationAfter2);
        lastDequantization2->set_friendly_name("MaxPool");
        results.push_back(std::make_shared<ngraph::opset1::Result>(lastDequantization2));
    }

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithDifferentChildsTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithIntermediateWithConstant(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const ngraph::element::Type precisionAfterDequantization) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, precisionBeforeOp);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, precisionBeforeOp);

    std::shared_ptr<Node> intermediateOp;

    if (transparentIntermediate) {
        const auto deqBefore = makeDequantization(fakeQuantize1->output(0), dequantizationBefore);
        const auto pooling = makeMaxPool(fakeQuantize1->output(0), { 3, 3 });

        ngraph::op::v0::InterpolateAttrs attributes;
        attributes.axes = ngraph::AxisSet{ 2, 3 };
        attributes.mode = "nearest";
        attributes.align_corners = false;
        attributes.antialias = false;
        attributes.pads_begin = { 0 };
        attributes.pads_end = { 0 };

        const auto outputShape = op::Constant::create(
            ngraph::element::i64, ngraph::Shape{ 2 },
            ngraph::Shape{ inputShape[2], inputShape[3] });
        intermediateOp = std::make_shared<ngraph::opset1::Interpolate>(pooling->output(0), outputShape, attributes);
        intermediateOp->set_friendly_name("intermediate");
    } else {
        intermediateOp = fakeQuantize1;
    }

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize2->output(0), intermediateOp->output(0) },
        1);
    concat->set_friendly_name("concat");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto deqAfter = makeDequantization(concat->output(0), dequantizationAfter);
    deqAfter->set_friendly_name("concat");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(deqAfter)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithReshapeAtTheEndTransformation(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const FakeQuantizeOnDataWithConstant& fqOnData3,
    const ngraph::element::Type precisionBeforeOp,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ngraph::opset1::Concat> concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    low_precision::NetworkHelper::setOutDataPrecision(concat1, precisionAfterOperation);
    concat1->set_friendly_name("concat1");

    std::shared_ptr<Node> intermediate = makeMaxPool(concat1->output(0), {1ul, 1ul});

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");

    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const std::shared_ptr<ngraph::opset1::Concat> concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ fakeQuantize3, intermediate }, 1);
    low_precision::NetworkHelper::setOutDataPrecision(concat2, precisionAfterOperation);
    concat2->set_friendly_name("concat2");

    const Shape concat2Shape = concat2->output(0).get_shape();
    const std::shared_ptr<Node> maxPool = makeMaxPool(concat2->output(0), {concat2Shape[2], concat2Shape[3]});
    const std::shared_ptr<Node> reshape = std::make_shared<ngraph::opset1::Reshape>(
        maxPool,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2ul}, std::vector<size_t>{0, 0}),
        true);
    reshape->set_friendly_name("output_original");

    const auto dequantization = makeDequantization(reshape->output(0), dequantizationOperations);
    dequantization->set_friendly_name("output");

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(dequantization)};

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2, input3 },
        "ReferenceWithReshapeAtTheEndTransformation");

    return function;
}

std::shared_ptr<Node> ConcatFunction::makeMaxPool(const Output<Node>& parent, const std::vector<size_t>& kernel) {
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;
    const auto pooling = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    return pooling;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph

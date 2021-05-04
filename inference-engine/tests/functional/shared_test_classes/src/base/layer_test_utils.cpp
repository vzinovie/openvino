// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include <transformations/serialize.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>
#include <ngraph/opsets/opset.hpp>
#include <pugixml.hpp>
#include <common_test_utils/file_utils.hpp>
#include <thread>

#include "ngraph/variant.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/core_config.hpp"
#include "ngraph/pass/visualize_tree.hpp"

namespace LayerTestsUtils {

LayerTestsCommon::LayerTestsCommon() : threshold(1e-2f) {
    core = PluginCache::get().ie(targetDevice);
}

void LayerTestsCommon::Run() {
    auto crashHandler = [](int errCode) {
        auto &s = Summary::getInstance();
        s.saveReport();
        std::cout << "Unexpected application crash!" << std::endl;
        std::abort();
    };
    signal(SIGSEGV, crashHandler);

    auto &s = Summary::getInstance();
    s.setDeviceName(targetDevice);

    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        s.updateOPsStats(function, PassRate::Statuses::SKIPPED);
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
    } else {
        s.updateOPsStats(function, PassRate::Statuses::CRASHED);
    }

    try {
        LoadNetwork();
        GenerateInputs();
        Infer();
        Validate();
        s.updateOPsStats(function, PassRate::Statuses::PASSED);
    }
    catch (const std::runtime_error &re) {
        s.updateOPsStats(function, PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_(re.what());
    } catch (const std::exception &ex) {
        s.updateOPsStats(function, PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_(ex.what());
    } catch (...) {
        s.updateOPsStats(function, PassRate::Statuses::FAILED);
        GTEST_FATAL_FAILURE_("Unknown failure occurred.");
    }
}

void LayerTestsCommon::Serialize() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    std::string output_name = GetTestName().substr(0, maxFileNameLength) + "_" + GetTimestamp();

    std::string out_xml_path = output_name + ".xml";
    std::string out_bin_path = output_name + ".bin";

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(out_xml_path, out_bin_path);
    manager.run_passes(function);

    auto result = getCore()->ReadNetwork(out_xml_path, out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
            compare_functions(result.getFunction(), function, false, false, false,
                              true,     // precision
                              true);    // attributes

    EXPECT_TRUE(success) << message;

    CommonTestUtils::removeIRFiles(out_xml_path, out_bin_path);
}

InferenceEngine::Blob::Ptr LayerTestsCommon::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}
void LayerTestsCommon::Compare(const std::vector<std::vector<std::uint8_t>> &expectedOutputs,
                               const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs,
                               float threshold) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto &expected = expectedOutputs[outputIndex];
        const auto &actual = actualOutputs[outputIndex];
        Compare(expected, actual, threshold);
    }
}

void LayerTestsCommon::Compare(const std::vector<std::uint8_t> &expected,
                               const InferenceEngine::Blob::Ptr &actual,
                               float threshold) {
    ASSERT_EQ(expected.size(), actual->byteSize());
    const auto &expectedBuffer = expected.data();

    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

    const auto &precision = actual->getTensorDesc().getPrecision();
    const auto &size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            Compare<float>(reinterpret_cast<const float *>(expectedBuffer),
                           reinterpret_cast<const float *>(actualBuffer), size, threshold);
            break;
        case InferenceEngine::Precision::I32:
            Compare<int32_t>(reinterpret_cast<const int32_t *>(expectedBuffer),
                             reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::I64:
            Compare<int64_t>(reinterpret_cast<const int64_t *>(expectedBuffer),
                             reinterpret_cast<const int64_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::I8:
            Compare<int8_t>(reinterpret_cast<const int8_t *>(expectedBuffer),
                            reinterpret_cast<const int8_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::U16:
            Compare<uint16_t>(reinterpret_cast<const uint16_t *>(expectedBuffer),
                              reinterpret_cast<const uint16_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::I16:
            Compare<int16_t>(reinterpret_cast<const int16_t *>(expectedBuffer),
                             reinterpret_cast<const int16_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::U8:
            Compare<uint8_t>(reinterpret_cast<const uint8_t *>(expectedBuffer),
                             reinterpret_cast<const uint8_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::U64:
            Compare<uint64_t>(reinterpret_cast<const uint64_t *>(expectedBuffer),
                              reinterpret_cast<const uint64_t *>(actualBuffer), size, 0);
            break;
        case InferenceEngine::Precision::BF16:
            Compare(reinterpret_cast<const ngraph::bfloat16 *>(expectedBuffer),
                    reinterpret_cast<const ngraph::bfloat16 *>(actualBuffer), size, ngraph::bfloat16(threshold));
            break;
        case InferenceEngine::Precision::FP16:
            Compare(reinterpret_cast<const ngraph::float16 *>(expectedBuffer),
                    reinterpret_cast<const ngraph::float16 *>(actualBuffer), size, ngraph::float16(threshold));
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void LayerTestsCommon::Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual) {
    Compare(expected, actual, threshold);
}

void LayerTestsCommon::Compare(const InferenceEngine::Blob::Ptr &expected, const InferenceEngine::Blob::Ptr &actual) {
    auto get_raw_buffer = [](const InferenceEngine::Blob::Ptr &blob) {
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        return lockedMemory.as<const std::uint8_t *>();
    };
    const auto expectedBuffer = get_raw_buffer(expected);
    const auto actualBuffer = get_raw_buffer(actual);

    const auto &precision = actual->getTensorDesc().getPrecision();
    const auto &size = actual->size();
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            Compare(reinterpret_cast<const float *>(expectedBuffer), reinterpret_cast<const float *>(actualBuffer),
                    size, threshold);
            break;
        case InferenceEngine::Precision::I32:
            Compare(reinterpret_cast<const std::int32_t *>(expectedBuffer),
                    reinterpret_cast<const std::int32_t *>(actualBuffer), size, 0);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}

void LayerTestsCommon::ConfigureNetwork() {
    for (const auto &in : cnnNetwork.getInputsInfo()) {
        if (inLayout != InferenceEngine::Layout::ANY) {
            in.second->setLayout(inLayout);
        }
        if (inPrc != InferenceEngine::Precision::UNSPECIFIED) {
            in.second->setPrecision(inPrc);
        }
    }

    for (const auto &out : cnnNetwork.getOutputsInfo()) {
        if (outLayout != InferenceEngine::Layout::ANY) {
            out.second->setLayout(outLayout);
        }
        if (outPrc != InferenceEngine::Precision::UNSPECIFIED) {
            out.second->setPrecision(outPrc);
        }
    }
}

void LayerTestsCommon::LoadNetwork() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    CoreConfiguration(this);
    ConfigureNetwork();
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
}

void LayerTestsCommon::GenerateInputs() {
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;
        auto blob = GenerateInput(*info);
        inputs.push_back(blob);
    }
}

void LayerTestsCommon::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();

    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (int i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;
        auto blob = inputs[i];
        inferRequest.SetBlob(info->name(), blob);
    }
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
        inferRequest.SetBatch(batchSize);
    }
    inferRequest.Infer();
}

std::vector<std::vector<std::uint8_t>> LayerTestsCommon::CalculateRefs() {
    // nGraph interpreter does not support f16/bf16
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(function);
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::bf16, ngraph::element::Type_t::f32>().run_on_function(function);

    ngraph::pass::VisualizeTree("/home/vzinoviev/work/model_dumps/model_after_convert.dot").run_on_function(function);

    function->validate_nodes_and_infer_types();

    auto referenceInputs = std::vector<std::vector<std::uint8_t>>(inputs.size());
    auto refInputsTypes = std::vector<ngraph::element::Type>(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto &input = inputs[i];
        const auto &inputSize = input->byteSize();

        auto &referenceInput = referenceInputs[i];
        referenceInput.resize(inputSize);

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto buffer = lockedMemory.as<const std::uint8_t *>();
        std::copy(buffer, buffer + inputSize, referenceInput.data());

        refInputsTypes[i] = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(memory->getTensorDesc().getPrecision());
    }

    const auto &&outputsInfo = executableNetwork.GetOutputsInfo();
    std::vector<ngraph::element::Type_t> convertType;
    convertType.reserve(outputsInfo.size());
        for (const auto &output : outputsInfo) {
                convertType.push_back(
                    FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(
                        output.second->getTensorDesc().getPrecision()));
        }

    std::vector<std::vector<std::uint8_t>> expectedOutputs;
    switch (refMode) {
        case INTERPRETER: {
            expectedOutputs = ngraph::helpers::interpreterFunction(function, referenceInputs, refInputsTypes, convertType);
            break;
        }
        case CONSTANT_FOLDING: {
            const auto &foldedFunc = ngraph::helpers::foldFunction(function, referenceInputs, refInputsTypes);
            expectedOutputs = ngraph::helpers::getConstData(foldedFunc, convertType);
            break;
        }
        case IE: {
            // reference inference on device with other options and nGraph function has to be implemented here
            break;
        }
    }

    return expectedOutputs;
}

std::vector<InferenceEngine::Blob::Ptr> LayerTestsCommon::GetOutputs() {
    auto outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
        const auto &name = output.first;
        outputs.push_back(inferRequest.GetBlob(name));
    }
    return outputs;
}

void LayerTestsCommon::Compare(const std::vector<std::vector<std::uint8_t>> &expectedOutputs,
                               const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) {
    Compare(expectedOutputs, actualOutputs, threshold);
}

void LayerTestsCommon::Validate() {
    auto expectedOutputs = CalculateRefs();
    const auto &actualOutputs = GetOutputs();

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
    << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    Compare(expectedOutputs, actualOutputs);
}

std::string LayerTestsCommon::getRuntimePrecision(const std::string& layerName) {
    const auto execGraph = executableNetwork.GetExecGraphInfo();
    const auto function = execGraph.getFunction();

    for (const auto& op : function->get_ops()) {
        const auto name = op->get_friendly_name();
        if (name == layerName) {
            const auto& rtInfo = op->get_rt_info();
            const auto& it = rtInfo.find("runtimePrecision");

            IE_ASSERT(it != rtInfo.end()) << "Runtime precision is not found for node: " << name;

            const auto rtPrecisionPtr = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(it->second);
            return rtPrecisionPtr->get();
        }
    }

    return "";
}

std::string LayerTestsCommon::getRuntimePrecisionByType(const std::string& layerType) {
    const auto execGraph = executableNetwork.GetExecGraphInfo();
    const auto function = execGraph.getFunction();

    for (const auto& op : function->get_ops()) {
        const auto& rtInfo = op->get_rt_info();
        const auto& typeIt = rtInfo.find("layerType");

        IE_ASSERT(typeIt != rtInfo.end()) << "Layer is not found for type: " << layerType;

        const auto type = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(typeIt->second)->get();
        if (type == layerType) {
            const auto& it = rtInfo.find("runtimePrecision");

            IE_ASSERT(it != rtInfo.end()) << "Runtime precision is not found for node: " << type;

            const auto rtPrecisionPtr = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(it->second);
            return rtPrecisionPtr->get();
        }
    }

    return "";
}

void LayerTestsCommon::SetRefMode(RefMode mode) {
    refMode = mode;
}

std::shared_ptr<ngraph::Function> LayerTestsCommon::GetFunction() {
    return function;
}

std::map<std::string, std::string> &LayerTestsCommon::GetConfiguration() {
    return configuration;
}

}  // namespace LayerTestsUtils

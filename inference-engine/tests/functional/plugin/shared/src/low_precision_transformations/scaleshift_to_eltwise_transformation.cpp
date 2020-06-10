// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/low_precision_transformations/scaleshift_to_eltwise_transformation.hpp"

namespace LayerTestsDefinitions {

std::string ScaleShiftToEltwiseTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << toString(params);
    return result.str();
}


void ScaleShiftToEltwiseTransformation::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto paramNode = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(inputShape));
    const auto fakeQuantize = makeFakeQuantize(paramNode->output(0));

    const std::vector<size_t> axisVector{ 0, 0, inputShape[2] / 2, 2, inputShape[3] / 2, 2 };
    const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::u64, ngraph::Shape{ axisVector.size() }, axisVector);
    const auto reshape = std::make_shared<ngraph::opset1::Reshape>(fakeQuantize->output(0), axes->output(0), true);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape) };
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ paramNode }, "ScaleShiftToEltwiseTransformation");

    // TODO: move to some another place
    validate();
}

std::shared_ptr<ngraph::opset1::FakeQuantize> ScaleShiftToEltwiseTransformation::makeFakeQuantize(const ngraph::Output<ngraph::Node>& input) {
    auto inputLowConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 0.f });
    auto inputHighConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 256.f });
    auto outputLowConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 0.f });
    auto outputHighConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 256.f / 2.f });
    auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(input, inputLowConst, inputHighConst, outputLowConst, outputHighConst, 256ul);
    return fakeQuantize;
}

IE_SUPPRESS_DEPRECATED_START

void ScaleShiftToEltwiseTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    const InferenceEngine::CNNNetwork network = transform(params);

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    network.serialize("C:\\Users\\vzinovie\\Documents\\dldt\\sstoew.xml", "C:\\Users\\vzinovie\\Documents\\dldt\\sstoew.bin");
    EXPECT_TRUE(outputLayer != nullptr);
    //EXPECT_EQ("Eltwise", outputLayer->type);
}

IE_SUPPRESS_DEPRECATED_END

TEST_P(ScaleShiftToEltwiseTransformation, CompareWithRefImpl) {
    Run();

    if (targetDevice == std::string{ CommonTestUtils::DEVICE_GPU }) {
        PluginCache::get().reset();
    }
};

}  // namespace LayerTestsDefinitions

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/low_precision_transformations/scaleshift_to_eltwise_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<InferenceEngine::details::LayerTransformation::Params> trasformationParamValues = {
        LayerTestsUtils::LayerTransformationParamsFactory::createParamCpu(),
        LayerTestsUtils::LayerTransformationParamsFactory::createParamU8I8()
    };

    INSTANTIATE_TEST_CASE_P(LPT, ScaleShiftToEltwiseTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 256, 256 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(trasformationParamValues)),
        ScaleShiftToEltwiseTransformation::getTestCaseName);
}  // namespace

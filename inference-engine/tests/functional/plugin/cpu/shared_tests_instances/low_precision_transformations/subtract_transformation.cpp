// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/subtract_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParams().setUpdatePrecisions(true),
    LayerTestsUtils::LayerTransformationParamsFactory::createParams().setUpdatePrecisions(false),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

INSTANTIATE_TEST_CASE_P(DISABLED_smoke_LPT, SubtractTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    SubtractTransformation::getTestCaseName);
}  // namespace
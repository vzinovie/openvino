// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

    class ScaleShiftToEltwiseTransformation : public LayerTestsUtils::LayerTransformation<LayerTestsUtils::LayerTransformationParams> {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj);

    protected:
        void SetUp() override;

    private:
        std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(const ngraph::Output<ngraph::Node>& output);
        void validate();
    };

}  // namespace LayerTestsDefinitions

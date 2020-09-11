// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/avg_pool_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(values.lowPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, originalFunctionPrecision);
    parent = convert;

    if (!values.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.subtractValues.size() }), values.subtractValues));
        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.mutliplyValues.size() }), values.mutliplyValues));
    parent = multiply;

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);

    std::shared_ptr<Node> lastLayer = avgPool;

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, originalFunctionPrecision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastLayer) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, ngraph::Shape(inputShape));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        fakeQuantize,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(avgPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ngraph::Function> AvgPoolFunction::getReference(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const ExpectedValues& values) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(values.activationPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::AvgPool>>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);
    const auto avgPoolPrecision = addFQ ? originalFunctionPrecision : values.activationPrecision;
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(avgPool, avgPoolPrecision);

    parent = avgPool;

    if (avgPoolPrecision != originalFunctionPrecision) {
        const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, originalFunctionPrecision);
        parent = convert;
    }

    if (!values.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.subtractValues.size() }), values.subtractValues));

        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Multiply>>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.mutliplyValues.size() }), values.mutliplyValues));

    std::shared_ptr<Node> lastLayer = multiply;

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, originalFunctionPrecision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastLayer) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph

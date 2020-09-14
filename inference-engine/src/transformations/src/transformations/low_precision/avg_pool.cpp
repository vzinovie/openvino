﻿// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/avg_pool.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

AvgPoolTransformation::AvgPoolTransformation(const Params& params) : LayerTransformation(params) {
}

void AvgPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::AvgPool>({ make_op_label<opset1::Multiply>() }));
}

bool AvgPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> pooling = separateInStandaloneBranch(m.get_match_root());

    const auto children = pooling->get_output_target_inputs(0);
    bool updatePrecision = false;
    // NOTE: This check was added for models that don't have FQ after AvgPool
    //       They will have transparent precision as it was in old LPT.
    for (const auto& child : children) {
        if (!is_type<opset1::FakeQuantize>(child.get_node()->shared_from_this())) {
            updatePrecision = true;
            break;
        }
    }
    moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling), updatePrecision);
    return true;
}

bool AvgPoolTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    auto dequantization = NetworkHelper::getDequantization(operation);

    return !!dequantization.multiply;
}

bool AvgPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    const auto children = layer->get_output_target_inputs(0);
    for (const auto& child : children) {
        if (!is_type<opset1::FakeQuantize>(child.get_node()->shared_from_this())) {
            return true;
        }
    }
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph

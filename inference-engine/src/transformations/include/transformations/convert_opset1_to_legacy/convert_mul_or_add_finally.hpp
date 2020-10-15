// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "ngraph_ops/scaleshift.hpp"
#include "ngraph_ops/eltwise.hpp"
#include "ngraph_ops/power.hpp"

#include "transformations/utils/utils.hpp"

#include "transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertMulOrAddFinally;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulOrAddFinally: public ngraph::pass::GraphRewrite {
public:
    // This pass finally converts single Multiply and Add operations to ScaleShift or Power operation
    ConvertMulOrAddFinally() : GraphRewrite() {
        convert_mul_or_add_finally<ngraph::opset1::Add>();
        convert_mul_or_add_finally<ngraph::opset1::Subtract>();
        convert_mul_or_add_finally<ngraph::opset1::Multiply>();
    }

private:
    template<typename T>
    void convert_mul_or_add_finally();
};

template <typename T>
bool convert_to_eltwise(std::shared_ptr<T> & node,
                        ngraph::Output<ngraph::Node> data1,
                        ngraph::Output<ngraph::Node> data2) {
    ELTWISE_TYPE et;
    if (std::is_same<T, ngraph::opset1::Multiply>()) {
        et = ELTWISE_TYPE::Prod;
    } else if (std::is_same<T, ngraph::opset1::Add>()) {
        et = ELTWISE_TYPE::Sum;
    } else if (std::is_same<T, ngraph::opset1::Subtract>()) {
        et = ELTWISE_TYPE::Sub;
    } else {
        return false;
    }

    auto eltwise = std::make_shared<ngraph::op::Eltwise>(data1, data2, et, node->output(0).get_element_type());
    eltwise->set_friendly_name(node->get_friendly_name());
    ngraph::copy_runtime_info(node, eltwise);
    ngraph::replace_node(node, eltwise);
    return true;
}

template <typename T>
ngraph::graph_rewrite_callback get_callback() {
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        static_assert(std::is_same<T, ngraph::opset1::Add>() || std::is_same<T, ngraph::opset1::Subtract>() || std::is_same<T, ngraph::opset1::Multiply>(),
                      "Unsupported template parameter. Only Add or Multiply allowed!");

        auto lin_op = std::dynamic_pointer_cast<T> (m.get_match_root());
        if (!lin_op || lin_op->output(0).get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        const auto output_shape = lin_op->output(0).get_partial_shape();
        const auto output_shape_rank = output_shape.rank().get_length();

        const auto intInputs = !lin_op->get_input_element_type(0).is_real() &&
                               !lin_op->get_input_element_type(1).is_real();

        if (!lin_op->get_element_type().is_real() || intInputs) {
            return convert_to_eltwise<T>(lin_op,
                                         lin_op->input(0).get_source_output(),
                                         lin_op->input(1).get_source_output());
        }

        std::shared_ptr<ngraph::opset1::Constant> const_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(
                lin_op->input(0).get_source_output().get_node_shared_ptr());
        auto data_node = lin_op->input(1).get_source_output();
        if (!const_node) {
            const_node = std::dynamic_pointer_cast<ngraph::opset1::Constant> (lin_op->input(1).get_source_output().get_node_shared_ptr());
            data_node = lin_op->input(0).get_source_output();
            if (!const_node) {
                return convert_to_eltwise<T>(lin_op,
                                             lin_op->input(0).get_source_output(),
                                             lin_op->input(1).get_source_output());
            }
        }

        /* This lambda checks data and constant shapes for broadcasting
           For example:
                1. data_shape{1, 64, 64} and const_shape{64, 1, 1} - constant broadcasts data_shape zero dimension
                2. data_shape{DYN, 64, 64} and const_shape{1, 1, 64} - constant do not broadcasts data_shape
                3. data_shape{64, 64} and const_shape{1, 1, 1} - constant broadcasts data_shape with additional dimension
        */
        auto constant_broadcast_output = [](const ngraph::PartialShape & data_pshape, const ngraph::Shape & const_shape) -> bool {
            if (data_pshape.rank().is_dynamic() || const_shape.size() > data_pshape.rank().get_length()) {
                return true;
            }

            std::vector<ngraph::Dimension> data_shape(data_pshape);

            auto const_shape_it = const_shape.rbegin();
            auto data_shape_it = data_shape.rbegin();

            while (const_shape_it != const_shape.rend()) {
                auto data_dim = *data_shape_it;
                auto const_dim = *const_shape_it;

                /* DATA DIM - CONST DIM - CONSTANT BROADCAST OUTPUT
                   DYN      - 64        - TRUE
                   DYN      - 1         - FALSE
                   64       - 1         - FALSE
                   1        - 64        - TRUE
                   64       - 64        - FALSE
                */
                if ((data_dim.is_dynamic() && const_dim != 1) ||
                    (data_dim.is_static() && data_dim.get_length() == 1 && const_dim != 1)) {
                    return true;
                }

                ++const_shape_it;
                ++data_shape_it;
            }

            return false;
        };

        // Check that eltwise is not useless and do not broadcast output otherwise we remove it
        if (((std::is_same<T, ngraph::opset1::Add>() && ngraph::op::util::constantIsEqualTo(const_node, 0)) ||
            (std::is_same<T, ngraph::opset1::Multiply>() && ngraph::op::util::constantIsEqualTo(const_node, 1))) &&
            !constant_broadcast_output(data_node.get_partial_shape(), const_node->get_shape())) {
            bool ret_status = ngraph::replace_output_update_name(lin_op->output(0), data_node);
            if (ret_status) {
                return true;
            }
        }

        auto res = check_constant(const_node, data_node.get_partial_shape());

        bool is_dequantization = lin_op->get_rt_info().count("DEQUANTIZATION") != 0;

        if (!is_dequantization && (res == CONVERSION_RESULT::NONE || (res == CONVERSION_RESULT::SCALE_SHIFT && output_shape_rank < 4))) {
            return convert_to_eltwise<T>(lin_op,
                                         lin_op->input(0).get_source_output(),
                                         lin_op->input(1).get_source_output());
        }

        // TODO: if all values in Constant are equal the best way is to convert this Eltwise to Power
        if (res == CONVERSION_RESULT::SCALE_SHIFT || is_dequantization) {
            auto weights_et = const_node->get_element_type();
            auto weights_shape = const_node->get_shape();

            // In case of Add we create fake weights with 1, in case of Multiply we create fake bias with 0
            std::shared_ptr<ngraph::op::ScaleShiftIE> scaleshift;
            if (std::is_same<T, ngraph::opset1::Add>()) {
                auto weights = ngraph::opset1::Constant::create(weights_et, weights_shape, {1});
                auto weights_in = ngraph::op::util::normalize_constant(weights, output_shape);
                auto biases_in = ngraph::op::util::normalize_constant(const_node, output_shape);
                if (is_dequantization) {
                    const ngraph::Shape data_shape = data_node.get_shape();
                    ngraph::Shape broadcasted_shape = std::vector<size_t>(data_shape.size(), 1ul);
                    broadcasted_shape[1] = data_shape[1];

                    weights_in = ngraph::op::util::broadcastTo(weights_in, broadcasted_shape);
                    biases_in = ngraph::op::util::broadcastTo(biases_in, broadcasted_shape);
                }
                scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data_node, weights_in, biases_in);
            } else if (std::is_same<T, ngraph::opset1::Subtract>()) {
                std::shared_ptr<ngraph::Node> new_const_node = std::make_shared<ngraph::opset1::Multiply>(
                    ngraph::op::util::normalize_constant(const_node, output_shape),
                    ngraph::opset1::Constant::create(weights_et, ngraph::Shape{ 1 }, { -1 }));

                auto weights = ngraph::opset1::Constant::create(weights_et, weights_shape, {1});
                auto weights_in = ngraph::op::util::normalize_constant(weights, output_shape);
                auto biases_in = new_const_node;
                if (is_dequantization) {
                    const ngraph::Shape data_shape = data_node.get_shape();
                    ngraph::Shape broadcasted_shape = std::vector<size_t>(data_shape.size(), 1ul);
                    broadcasted_shape[1] = data_shape[1];

                    weights_in = ngraph::op::util::broadcastTo(weights_in, broadcasted_shape);
                    biases_in = ngraph::op::util::broadcastTo(biases_in, broadcasted_shape);
                }
                scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data_node, weights_in, biases_in);
            } else if (std::is_same<T, ngraph::opset1::Multiply>()) {
                auto bias = ngraph::opset1::Constant::create(weights_et, weights_shape, {0});
                auto weights_in = ngraph::op::util::normalize_constant(const_node, output_shape);
                auto biases_in = ngraph::op::util::normalize_constant(bias, output_shape);
                if (is_dequantization) {
                    const ngraph::Shape data_shape = data_node.get_shape();
                    ngraph::Shape broadcasted_shape = std::vector<size_t>(data_shape.size(), 1ul);
                    broadcasted_shape[1] = data_shape[1];

                    weights_in = ngraph::op::util::broadcastTo(weights_in, broadcasted_shape);
                    biases_in = ngraph::op::util::broadcastTo(biases_in, broadcasted_shape);
                }
                scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data_node, weights_in, biases_in);
            } else {
                return false;
            }

            scaleshift->set_friendly_name(lin_op->get_friendly_name());
            ngraph::copy_runtime_info(m.get_match_root(), scaleshift);
            ngraph::replace_node(m.get_match_root(), scaleshift);
        } else {
            float value;
            if (!ngraph::op::util::get_single_value(const_node, value)) {
                return false;
            }

            // In case Add we create fake scale equal to 1, in case of Multiply we create fake shift equal to 0
            std::shared_ptr<ngraph::op::PowerIE> power;
            if (std::is_same<T, ngraph::opset1::Add>()) {
                power = std::make_shared<ngraph::op::PowerIE>(data_node, 1., 1., value, lin_op->get_output_element_type(0));
            } else if (std::is_same<T, ngraph::opset1::Multiply>()) {
                power = std::make_shared<ngraph::op::PowerIE>(data_node, 1., value, 0., lin_op->get_output_element_type(0));
            } else if (std::is_same<T, ngraph::opset1::Subtract>()) {
                power = std::make_shared<ngraph::op::PowerIE>(data_node, 1., 1., -value, lin_op->get_output_element_type(0));
            } else {
                return false;
            }
            power->set_friendly_name(lin_op->get_friendly_name());
            ngraph::copy_runtime_info(m.get_match_root(), power);
            ngraph::replace_node(m.get_match_root(), power);
        }

        return true;
    };
    return callback;
}

template <typename T>
void ngraph::pass::ConvertMulOrAddFinally::convert_mul_or_add_finally() {
    auto data_batch_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});
    auto data_batch_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});

    auto lin_op = std::make_shared<T>(data_batch_1, data_batch_2);

    auto m = std::make_shared<ngraph::pattern::Matcher>(lin_op);
    this->add_matcher(m, get_callback<T>(), PassProperty::CHANGE_DYNAMIC_STATE);
}

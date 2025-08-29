// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <pybind11/pybind11.h>

// Standardized macro for raising Python exceptions: log, set PyErr, then throw
#define PY_THROW_WITH_LOG(PyExcType, MessageExpr)      \
  do {                                                 \
    const std::string& _sc_err_msg = (MessageExpr);    \
    LOG(ERROR) << _sc_err_msg;                         \
    PyErr_SetString((PyExcType), _sc_err_msg.c_str()); \
    throw pybind11::error_already_set();               \
  } while (0)

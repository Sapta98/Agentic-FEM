import macro from 'vtk.js/Sources/macros';
/**
 * Scalar color mixin.
 * Scalar value [0, 1] references a color in the mapper LUT.
 * Not to be used in conjunction with `color3` mixin.
 * @see color3
 */

const DEFAULT_VALUES = {
  color: 0.5,
};

// ----------------------------------------------------------------------------

export function extend(publicAPI, model, initialValues = {}) {
  Object.assign(model, DEFAULT_VALUES, initialValues);
  macro.setGet(publicAPI, model, ['color']);
}

// ----------------------------------------------------------------------------

export default { extend };

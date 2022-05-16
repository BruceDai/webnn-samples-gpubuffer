'use strict';

import {BaseNetwork} from '../common/base_net_polyfill.js';
import {buildConstantByNpy} from '../common/utils.js';

// MobileNet V2 model with 'nchw' input layout
export class MobileNetV2NchwPolyfill extends BaseNetwork {
  constructor() {
    super();
    this.weightsUrl_ = '../test-data/models/mobilenetv2_nchw/weights/';
    this.inputOptions = {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      norm: true,
      inputLayout: 'nchw',
      labelUrl: './labels/labels1000.txt',
      inputDimensions: [1, 3, 224, 224],
    };
    this.outputDimensions = [1, 1000];
  }

  async buildConv_(input, name, relu6 = true, options = {}) {
    const prefix = this.weightsUrl_ + 'conv_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights =
        await buildConstantByNpy(this.device_, this.builder_, weightsName, true);
    const biasName = prefix + '_bias.npy';
    const bias =
        await buildConstantByNpy(this.device_, this.builder_, biasName, true);
    options.bias = bias;
    if (relu6) {
      // implement `relu6` by `clamp` of  WebNN API
      options.activation = this.builder_.clamp({minValue: 0, maxValue: 6});
    } else {
      options.activation = undefined;
    }
    return this.builder_.conv2d(input, weights, options);
  }

  async buildGemm_(input, name) {
    const prefix = this.weightsUrl_ + 'gemm_' + name;
    const weightsName = prefix + '_weight.npy';
    const weights = await buildConstantByNpy(this.device_, this.builder_, weightsName, true);
    const biasName = prefix + '_bias.npy';
    const bias = await buildConstantByNpy(this.device_, this.builder_, biasName, true);
    const options = {c: bias, bTranspose: true};
    return this.builder_.gemm(input, weights, options);
  }

  async buildLinearBottleneck_(
      input, convNameArray, group, stride, shortcut = true) {
    const conv1x1Relu6 = await this.buildConv_(input, convNameArray[0]);
    const options = {
      padding: [1, 1, 1, 1],
      groups: group,
      strides: [stride, stride],
    };
    const dwise3x3Relu6 = await this.buildConv_(
        conv1x1Relu6, convNameArray[1], true, options);
    const conv1x1Linear = await this.buildConv_(
        dwise3x3Relu6, convNameArray[2], false);

    if (shortcut) {
      return this.builder_.add(input, conv1x1Linear);
    }
    return conv1x1Linear;
  }

  async load(devicePreference) {
    await this.init();
    const context = navigator.ml_polyfill.createContext();
    this.builder_ = new MLGraphBuilderPolyfill(context);
    const data = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const conv0 = await this.buildConv_(
        data, '0', true, {padding: [1, 1, 1, 1], strides: [2, 2]});
    const conv1 = await this.buildConv_(
        conv0, '2', true, {padding: [1, 1, 1, 1], groups: 32});
    const conv2 = await this.buildConv_(conv1, '4', false);
    const bottleneck0 = await this.buildLinearBottleneck_(
        conv2, ['5', '7', '9'], 96, 2, false);
    const bottleneck1 = await this.buildLinearBottleneck_(
        bottleneck0, ['10', '12', '14'], 144, 1);
    const bottleneck2 = await this.buildLinearBottleneck_(
        bottleneck1, ['16', '18', '20'], 144, 2, false);
    const bottleneck3 = await this.buildLinearBottleneck_(
        bottleneck2, ['21', '23', '25'], 192, 1);
    const bottleneck4 = await this.buildLinearBottleneck_(
        bottleneck3, ['27', '29', '31'], 192, 1);
    const bottleneck5 = await this.buildLinearBottleneck_(
        bottleneck4, ['33', '35', '37'], 192, 2, false);
    const bottleneck6 = await this.buildLinearBottleneck_(
        bottleneck5, ['38', '40', '42'], 384, 1);
    const bottleneck7 = await this.buildLinearBottleneck_(
        bottleneck6, ['44', '46', '48'], 384, 1);
    const bottleneck8 = await this.buildLinearBottleneck_(
        bottleneck7, ['50', '52', '54'], 384, 1);
    const bottleneck9 = await this.buildLinearBottleneck_(
        bottleneck8, ['56', '58', '60'], 384, 1, false);
    const bottleneck10 = await this.buildLinearBottleneck_(
        bottleneck9, ['61', '63', '65'], 576, 1);
    const bottleneck11 = await this.buildLinearBottleneck_(
        bottleneck10, ['67', '69', '71'], 576, 1);
    const bottleneck12 = await this.buildLinearBottleneck_(
        bottleneck11, ['73', '75', '77'], 576, 2, false);
    const bottleneck13 = await this.buildLinearBottleneck_(
        bottleneck12, ['78', '80', '82'], 960, 1);
    const bottleneck14 = await this.buildLinearBottleneck_(
        bottleneck13, ['84', '86', '88'], 960, 1);
    const bottleneck15 = await this.buildLinearBottleneck_(
        bottleneck14, ['90', '92', '94'], 960, 1, false);

    const conv3 = await this.buildConv_(bottleneck15, '95', true);
    const pool = this.builder_.averagePool2d(conv3);
    const reshape = this.builder_.reshape(pool, [1, -1]);
    const gemm = await this.buildGemm_(reshape, '104');
    return this.builder_.softmax(gemm);
  }
}

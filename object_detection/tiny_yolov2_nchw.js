'use strict';

import {buildConstantByNpy, sizeOfShape, createGPUBuffer} from '../common/utils.js';

// Tiny Yolo V2 model with 'nchw' layout, trained on the Pascal VOC dataset.
export class TinyYoloV2Nchw {
  constructor() {
    this.device_ = null;
    this.builder_ = null;
    this.graph_ = null;
    this.weightsUrl_ = '../test-data/models/tiny_yolov2_nchw/weights/';
    this.inputOptions = {
      inputLayout: 'nchw',
      labelUrl: './labels/pascal_classes.txt',
      margin: [1, 1, 1, 1],
      anchors: [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
      inputDimensions: [1, 3, 416, 416],
    };
    this.outputDimensions = [1, 125, 13, 13];
    this.inputSizeInBytes_ = sizeOfShape(this.inputOptions.inputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.outputSizeInBytes_ = sizeOfShape(this.outputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.inputGPUBuffers_ = [];
    this.outputGPUBuffer_ = null;
  }

  async buildConv_(input, name, useBias = false) {
    const prefix = this.weightsUrl_ + 'convolution' + name;
    const weightName = prefix + '_W.npy';
    const weight = await buildConstantByNpy(this.device_, this.builder_, weightName);
    const options = {autoPad: 'same-upper'};
    if (useBias) {
      const biasName = prefix + '_B.npy';
      options.bias = await buildConstantByNpy(this.device_, this.builder_, biasName);
    }
    return this.builder_.conv2d(input, weight, options);
  }

  async buildBatchNorm_(input, name) {
    const prefix = this.weightsUrl_ + 'BatchNormalization';
    const scaleName = `${prefix}_scale${name}.npy`;
    const biasName = `${prefix}_B${name}.npy`;
    const meanName = `${prefix}_mean${name}.npy`;
    const varName = `${prefix}_variance${name}.npy`;
    const scale = await buildConstantByNpy(this.device_, this.builder_, scaleName);
    const bias = await buildConstantByNpy(this.device_, this.builder_, biasName);
    const mean = await buildConstantByNpy(this.device_, this.builder_, meanName);
    const variance = await buildConstantByNpy(this.device_, this.builder_, varName);

    const batchNorm = this.builder_.batchNormalization(
        input, mean, variance, {scale: scale, bias: bias,
          activation: this.builder_.leakyRelu({alpha: 0.10000000149011612})});
    return batchNorm;
  }

  async buildConvolutional_(input, name) {
    const conv = await this.buildConv_(input, name);
    return await this.buildBatchNorm_(conv, name);
  }

  async load(devicePreference) {
    const adaptor = await navigator.gpu.requestAdapter();
    this.device_ = await adaptor.requestDevice();
    const context = navigator.ml.createContext(this.device_);
    this.builder_ = new MLGraphBuilder(context);
    const image = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});

    const mulScale = this.builder_.constant(
        { type: 'float32', dimensions: [1] },
        { resource: await createGPUBuffer(this.device_, sizeOfShape([1]), [0.003921568859368563]) });
    const addBias = this.builder_.constant(
        { type: 'float32', dimensions: [3, 1, 1] },
        { resource: await createGPUBuffer(this.device_, sizeOfShape([3, 1, 1]), [0, 0, 0]) });
    const poolOptions = {
      windowDimensions: [2, 2],
      strides: [2, 2],
      autoPad: 'same-upper',
    };
    const mul = this.builder_.mul(image, mulScale);
    const add = this.builder_.add(mul, addBias);
    const conv0 = await this.buildConvolutional_(add, '');
    const pool0 = this.builder_.maxPool2d(conv0, poolOptions);
    const conv1 = await this.buildConvolutional_(pool0, '1');
    const pool1 = this.builder_.maxPool2d(conv1, poolOptions);
    const conv2 = await this.buildConvolutional_(pool1, '2');
    const pool2 = this.builder_.maxPool2d(conv2, poolOptions);
    const conv3 = await this.buildConvolutional_(pool2, '3');
    const pool3 = this.builder_.maxPool2d(conv3, poolOptions);
    const conv4 = await this.buildConvolutional_(pool3, '4');
    const pool4 = this.builder_.maxPool2d(conv4, poolOptions);
    const conv5 = await this.buildConvolutional_(pool4, '5');
    const pool5 = this.builder_.maxPool2d(conv5,
        {windowDimensions: [2, 2], autoPad: 'same-upper'});
    const conv6 = await this.buildConvolutional_(pool5, '6');
    const conv7 = await this.buildConvolutional_(conv6, '7');
    const conv = await this.buildConv_(conv7, '8', true);
    return conv;
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
    this.outputGPUBuffer_ = this.device_.createBuffer({size: this.outputSizeInBytes_, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  }

  // Release the constant tensors of a model
  dispose() {
    // dispose() is only available in webnn-polyfill
    if (this.graph_ !== null && 'dispose' in this.graph_) {
      this.graph_.dispose();
    }
  }

  async compute(inputBuffer, outputs) {
    let inputGPUBuffer;
    if (this.inputGPUBuffers_.length) {
      inputGPUBuffer = this.inputGPUBuffers_.pop();
    } else {
      inputGPUBuffer = this.device_.createBuffer({
        size: this.inputSizeInBytes_,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
      });
      await inputGPUBuffer.mapAsync(GPUMapMode.WRITE);
    }
    new Float32Array(inputGPUBuffer.getMappedRange()).set(inputBuffer);
    inputGPUBuffer.unmap();
    this.graph_.compute({'input': {resource: inputGPUBuffer}}, {'output': {resource: this.outputGPUBuffer_}});
    inputGPUBuffer.mapAsync(GPUMapMode.WRITE).then(() => {
      this.inputGPUBuffers_.push(inputGPUBuffer);
    });
    await this.outputGPUBuffer_.mapAsync(GPUMapMode.READ);
    outputs.output.set(new Float32Array(this.outputGPUBuffer_.getMappedRange()));
    this.outputGPUBuffer_.unmap();
  }
}

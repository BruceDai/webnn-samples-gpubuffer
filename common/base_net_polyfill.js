import { sizeOfShape } from "./utils.js";

export class BaseNetwork {
  constructor() {
    this.device_ = null;
    this.builder_ = null;
    this.graph_ = null;
  }
  async init() {
    await tf.setBackend('webgpu');
    this.device_ = tf.engine().backendInstance.device;
    this.inputSizeInBytes_ = sizeOfShape(this.inputOptions.inputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.outputSizeInBytes_ = sizeOfShape(this.outputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.inputGPUBuffers_ = [];
    this.outputGPUBuffer_ = null;
    this.outputGPUBufferForProcessing_ = null;
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
    this.outputGPUBufferForProcessing_ = {resource: null};
    if (this.inputOptions.std) {
      this.inputOptions.stdTensor = tf.tensor1d(this.inputOptions.std);
    }
    if (this.inputOptions.mean) {
      this.inputOptions.meanTensor = tf.tensor1d(this.inputOptions.mean);
    }
    if (this.inputOptions.norm) {
      this.inputOptions.normTensor = tf.tensor1d([255, 255, 255]);
    }
  }

  // dispose() {
  //   // dispose() is only available in webnn-polyfill
  //   if (this.graph_ !== null && 'dispose' in this.graph_) {
  //     this.graph_.dispose();
  //   }
  //   if (this.inputOptions.meanTensor instanceof tf.Tensor) {
  //     this.inputOptions.meanTensor.dispose();
  //   }
  //   if (this.inputOptions.stdTensor instanceof tf.Tensor) {
  //     this.inputOptions.stdTensor.dispose();
  //   }
  //   if (this.inputOptions.normTensor instanceof tf.Tensor) {
  //     this.inputOptions.normTensor.dispose();
  //   }
  // }

  async computeGPUTensor(inputTensor, outputBuffer, typedArrayConstructor = Float32Array) {
    const outputResource = {resource: null};
    this.graph_.compute({'input': {resource: inputTensor}}, {'output': outputResource});
    outputBuffer.set(await outputResource.resource.data());
    outputResource.resource.dispose();
  }

  computeOutputGPUTensor(inputTensor) {
    const outputResource = {resource: null};
    this.graph_.compute({'input': {resource: inputTensor}}, {'output': outputResource});
    return outputResource.resource;
  }

  async computeGPUTensorToGPUBuffer(inputTensor) {
    this.graph_.compute({'input': {resource: inputTensor}}, {'output': this.outputGPUBufferForProcessing_});
  }

  getOutputGPUBufferForProcessing() {
    return this.outputGPUBufferForProcessing_.resource;
  }

  async compute(inputBuffer, outputBuffer, typedArrayConstructor = Float32Array) {
    let inputGPUBuffer;
    if (this.inputGPUBuffers_.length) {
      inputGPUBuffer = this.inputGPUBuffers_.pop();
    } else {
      console.log('create buffer');
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
    outputBuffer.set(new typedArrayConstructor(this.outputGPUBuffer_.getMappedRange()));
    this.outputGPUBuffer_.unmap();
  }
}
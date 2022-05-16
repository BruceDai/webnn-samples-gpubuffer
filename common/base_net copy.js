import { sizeOfShape, createGPUBuffer, readbackGPUBuffer } from "./utils.js";

export class BaseNetwork {
  constructor() {
    this.device_ = null;
    this.builder_ = null;
    this.graph_ = null;
  }
  async init() {
    const adaptor = await navigator.gpu.requestAdapter();
    this.device_ = await adaptor.requestDevice();
    this.inputSizeInBytes_ = sizeOfShape(this.inputOptions.inputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.outputSizeInBytes_ = sizeOfShape(this.outputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.inputGPUBuffers_ = [];
    this.outputGPUBuffer_ = null;
    // this.outputGPUBufferForProcessing_ = null;
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
    this.outputGPUBuffer_ = this.device_.createBuffer({size: this.outputSizeInBytes_, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    // this.outputGPUBufferForProcessing_ = this.device_.createBuffer({size: this.outputSizeInBytes_, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE});
    // if (this.inputOptions.std) {
    //   this.inputOptions.stdTensor = this.builder_.constant(
    //     { type: 'float32', dimensions: [3] },
    //     { resource: await createGPUBuffer(this.device_, sizeOfShape([3]), this.inputOptions.std) });
    // }
    // if (this.inputOptions.mean) {
    //   this.inputOptions.meanTensor = this.builder_.constant(
    //     { type: 'float32', dimensions: [3] },
    //     { resource: await createGPUBuffer(this.device_, sizeOfShape([3]), this.inputOptions.mean) });
    // }
    // if (this.inputOptions.norm) {
    //   this.inputOptions.normTensor = this.builder_.constant(
    //     { type: 'float32', dimensions: [3] },
    //     { resource: await createGPUBuffer(this.device_, sizeOfShape([3]), [255, 255, 255]) });
    // }
  }

  async computeGPUTensor(inputData, inputDimensions, outputBuffer, typedArrayConstructor = Float32Array) {
    const inputGPUBuffer = await createGPUBuffer(this.device_, sizeOfShape(inputDimensions), inputData);
    await this.graph_.computeAsync({'input': {resource: inputGPUBuffer}}, {'output': {resource: this.outputGPUBuffer_}});
    // await this.outputGPUBuffer_.mapAsync(GPUMapMode.READ);
    // outputBuffer.set(new typedArrayConstructor(this.outputGPUBuffer_.getMappedRange()));
    // this.outputGPUBuffer_.unmap();

  }

  // // semantic segmentation
  // computeOutputGPUTensor(inputTensor) {
  //   const outputTensor = tf.tidy(() => tf.add(tf.zeros(this.outputDimensions), tf.zeros(this.outputDimensions)));
  //   tf.engine().backendInstance.submitQueue();
  //   const inputGPUBuffer = tf.engine().backendInstance.getBuffer(inputTensor.dataId);
  //   const outputGPUBuffer = tf.engine().backendInstance.getBuffer(outputTensor.dataId);
  //   await this.graph_.computeAsync({'input': {resource: inputGPUBuffer}}, {'output': {resource: outputGPUBuffer}});
  //   return outputTensor;
  // }

  // async computeGPUTensorToGPUBuffer(inputTensor) {
  //   const inputGPUBuffer = tf.engine().backendInstance.getBuffer(inputTensor.dataId);
  //   this.graph_.compute({'input': {resource: inputGPUBuffer}}, {'output': {resource: this.outputGPUBufferForProcessing_}});
  // }

  // getOutputGPUBufferForProcessing() {
  //   return this.outputGPUBufferForProcessing_;
  // }

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
    this.graph_.computeAsync({'input': {resource: inputGPUBuffer}}, {'output': {resource: this.outputGPUBuffer_}});
    inputGPUBuffer.mapAsync(GPUMapMode.WRITE).then(() => {
      this.inputGPUBuffers_.push(inputGPUBuffer);
    });
    await this.outputGPUBuffer_.mapAsync(GPUMapMode.READ);
    outputBuffer.set(new typedArrayConstructor(this.outputGPUBuffer_.getMappedRange()));
    this.outputGPUBuffer_.unmap();
  }
}
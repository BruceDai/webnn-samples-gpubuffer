'use strict';

import {MobileNetV2NchwPolyfill} from './mobilenet_nchw_polyfill.js';
import {MobileNetV2Nchw} from './mobilenet_nchw.js';
import {MobileNetV2Nhwc} from './mobilenet_nhwc.js';
import {SqueezeNetNchw} from './squeezenet_nchw.js';
import {SqueezeNetNhwc} from './squeezenet_nhwc.js';
import {ResNet50V2Nchw} from './resnet50v2_nchw.js';
import {ResNet50V2Nhwc} from './resnet50v2_nhwc.js';
import {ResNet101V2Nhwc} from './resnet101v2_nhwc.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';

const maxWidth = 380;
const maxHeight = 380;
const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
let modelName = '';
let layout = 'nchw';
let instanceType = modelName + layout;
let rafReq;
let isFirstTimeLoad = true;
let inputType = 'image';
let netInstance = null;
let labels = null;
let stream = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let inputOptions;
let outputBuffer;
let devicePreference = 'gpu';
let lastDevicePreference = '';
const disabledSelectors = ['#tabs > li', '.btn'];

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
}

$(document).ready(() => {
  $('.icdisplay').hide();
});

$('#deviceBtns .btn').on('change', async (e) => {
  devicePreference = $(e.target).attr('id');
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

$('#modelBtns .btn').on('change', async (e) => {
  modelName = $(e.target).attr('id');
  // Disable ResNet 101 NCHW as we haven't support it yet
  if (modelName === 'resnet101') {
    $('#nchw-label').addClass('disabled');
    $('#nchw-label').addClass('btn-outline-secondary');
    $('#nchw-label').removeClass('btn-outline-info');
  } else {
    $('#nchw-label').removeClass('disabled');
    $('#nchw-label').removeClass('btn-outline-secondary');
    $('#nchw-label').addClass('btn-outline-info');
  }
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

$('#layoutBtns .btn').on('change', async (e) => {
  layout = $(e.target).attr('id');
  // Disable ResNet 101 NCHW as we haven't support it yet
  if (layout === 'nhwc') {
    $('#resnet101-label').removeClass('disabled');
    $('#resnet101-label').removeClass('btn-outline-secondary');
    $('#resnet101-label').addClass('btn-outline-info');
  } else {
    $('#resnet101-label').addClass('disabled');
    $('#resnet101-label').addClass('btn-outline-secondary');
    $('#resnet101-label').removeClass('btn-outline-info');
  }
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera') cancelAnimationFrame(rafReq);
  if (stream !== null) stopCamera();
  inputType = 'image';
  $('.shoulddisplay').hide();
  await main();
});

$('#imageFile').change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $('#feedElement').removeAttr('height');
    $('#feedElement').removeAttr('width');
    imgElement.src = URL.createObjectURL(files[0]);
  }
});

$('#feedElement').on('load', async () => {
  await main();
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('.shoulddisplay').hide();
  await main();
});

async function getMediaStream() {
  // Support 'user' facing mode at present
  const constraints = {audio: false, video: {facingMode: 'user'}};
  stream = await navigator.mediaDevices.getUserMedia(constraints);
}

function stopCamera() {
  stream.getTracks().forEach((track) => {
    if (track.readyState === 'live' && track.kind === 'video') {
      track.stop();
    }
  });
}

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  // If the video element's readyState is 0, the video's width and height are 0.
  // So check the readState here to make sure it is greater than 0.
  if (camElement.readyState === 0) {
    rafReq = requestAnimationFrame(renderCamStream);
    return;
  }
  
  console.log('- Computing... ');
  const inputData = utils.getInputTensor(camElement, inputOptions);
  const inputGPUBuffer = await utils.createGPUBuffer(netInstance.device_, utils.sizeOfShape(inputOptions.inputDimensions), inputData); 
  const start = performance.now();
  await netInstance.compute(inputGPUBuffer, outputBuffer); // computeGPUTensor compute
  computeTime = (performance.now() - start).toFixed(2);
  console.log(`  done in ${computeTime} ms.`);
  // if (inputBuffer instanceof tf.Tensor) {
  //   inputBuffer.dispose();
  // }
  drawInput(camElement, 'camInCanvas');
  showPerfResult();
  await drawOutput(outputBuffer, labels);
  $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
  rafReq = requestAnimationFrame(renderCamStream);
}

// Get top 3 classes of labels from output buffer
function getTopClasses(buffer, labels) {
  const probs = Array.from(buffer);
  const indexes = probs.map((prob, index) => [prob, index]);
  const sorted = indexes.sort((a, b) => {
    if (a[0] === b[0]) {
      return 0;
    }
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();
  const classes = [];

  for (let i = 0; i < 3; ++i) {
    const prob = sorted[i][0];
    const index = sorted[i][1];
    const c = {
      label: labels[index],
      prob: (prob * 100).toFixed(2),
    };
    classes.push(c);
  }

  return classes;
}

function drawInput(srcElement, canvasId) {
  const inputCanvas = document.getElementById(canvasId);
  srcElement.width = srcElement.videoWidth ||
      srcElement.naturalWidth;
  srcElement.height = srcElement.videoHeight ||
      srcElement.naturalHeight;
  const resizeRatio = Math.max(
      Math.max(srcElement.width / maxWidth, srcElement.height / maxHeight), 1);
  const scaledWidth = Math.floor(srcElement.width / resizeRatio);
  const scaledHeight = Math.floor(srcElement.height / resizeRatio);
  inputCanvas.height = scaledHeight;
  inputCanvas.width = scaledWidth;
  const ctx = inputCanvas.getContext('2d');
  ctx.drawImage(srcElement, 0, 0, scaledWidth, scaledHeight);
}

async function drawOutput(outputBuffer, labels) {
  const labelClasses = getTopClasses(outputBuffer, labels);

  $('#inferenceresult').show();
  labelClasses.forEach((c, i) => {
    const labelElement = document.getElementById(`label${i}`);
    const probElement = document.getElementById(`prob${i}`);
    labelElement.innerHTML = `${c.label}`;
    probElement.innerHTML = `${c.prob}%`;
  });
}

function showPerfResult(medianComputeTime = undefined) {
  $('#loadTime').html(`${loadTime} ms`);
  $('#buildTime').html(`${buildTime} ms`);
  if (medianComputeTime !== undefined) {
    $('#computeLabel').html('Median inference time:');
    $('#computeTime').html(`${medianComputeTime} ms`);
  } else {
    $('#computeLabel').html('Inference time:');
    $('#computeTime').html(`${computeTime} ms`);
  }
}

function constructNetObject(type) {
  const netObject = {
    'mobilenetnchw': new MobileNetV2Nchw(),
    'mobilenetnchwpolyfill': new MobileNetV2NchwPolyfill(),
    'mobilenetnhwc': new MobileNetV2Nhwc(),
    'squeezenetnchw': new SqueezeNetNchw(),
    'squeezenetnhwc': new SqueezeNetNhwc(),
    'resnet50nchw': new ResNet50V2Nchw(),
    'resnet50nhwc': new ResNet50V2Nhwc(),
    'resnet101nhwc': new ResNet101V2Nhwc(),
  };

  return netObject[type];
}

async function main() {
  try {
    // await tf.setBackend('webgpu');
    // tf.env().set('WEBGPU_USE_IMPORT', true);
    if (modelName === '') return;
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $('#hint').hide();
    let start;
    // Set 'numRuns' param to run inference multiple times
    const params = new URLSearchParams(location.search);
    let numRuns = params.get('numRuns');
    numRuns = numRuns === null ? 1 : parseInt(numRuns);
    const isPolyfill = params.get('polyfill') === 'true';

    if (numRuns < 1) {
      ui.addAlert('The value of param numRuns must be greater than or equal' +
          ' to 1.');
      return;
    }
    // Only do load() and build() when model first time loads,
    // there's new model choosed, and device backend changed
    if (isFirstTimeLoad || instanceType !== modelName + layout ||
        lastDevicePreference != devicePreference) {
      if (lastDevicePreference != devicePreference) {
        // Set polyfill backend
        // await utils.setPolyfillBackend(devicePreference);
        lastDevicePreference = devicePreference;
      }
      // if (netInstance !== null) {
      //   // Call dispose() to and avoid memory leak
      //   netInstance.dispose();
      // }
      instanceType = modelName + layout + (isPolyfill ? 'polyfill' : '');
      console.log(instanceType);
      netInstance = constructNetObject(instanceType);
      inputOptions = netInstance.inputOptions;
      labels = await fetchLabels(inputOptions.labelUrl);
      outputBuffer =
          new Float32Array(utils.sizeOfShape(netInstance.outputDimensions));
      isFirstTimeLoad = false;
      console.log(`- Model name: ${modelName}, Model layout: ${layout} -`);
      // UI shows model loading progress
      await ui.showProgressComponent('current', 'pending', 'pending');
      console.log('- Loading weights... ');
      start = performance.now();
      const outputOperand = await netInstance.load(devicePreference);
      loadTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${loadTime} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent('done', 'current', 'pending');
      console.log('- Building... ');
      start = performance.now();
      await netInstance.build(outputOperand);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      const inputBuffer = utils.getInputTensor(imgElement, inputOptions);
      console.log('- Computing... ');
      const computeTimeArray = [];
      let medianComputeTime;

      // Do warm up compute
      let warmupstart = performance.now();
      await netInstance.compute(inputBuffer, outputBuffer); // computeGPUTensor
      let warmup = (performance.now() - warmupstart).toFixed(2);
      console.log(`  warmup time: ${warmup} ms`);

      // if (inputBuffer instanceof tf.Tensor) {
      //   inputBuffer.dispose();
      // }
      for (let i = 0; i < numRuns; i++) {
        const inputBuffer = utils.getInputTensor(imgElement, inputOptions);
        start = performance.now();
        await netInstance.compute(inputBuffer, outputBuffer); // computeGPUTensor compute
        computeTime = (performance.now() - start).toFixed(2);
        console.log(`  compute time ${i+1}: ${computeTime} ms`);
        computeTimeArray.push(Number(computeTime));
        // if (inputBuffer instanceof tf.Tensor) {
        //   inputBuffer.dispose();
        // }
      }
      if (numRuns > 1) {
        medianComputeTime = utils.getMedianValue(computeTimeArray);
        medianComputeTime = medianComputeTime.toFixed(2);
        console.log(`  median compute time: ${medianComputeTime} ms`);
      }
      console.log('outputBuffer: ', outputBuffer);
      await ui.showProgressComponent('done', 'done', 'done');
      ui.readyShowResultComponents();
      drawInput(imgElement, 'inputCanvas');
      await drawOutput(outputBuffer, labels);
      showPerfResult(medianComputeTime);
      
    } else if (inputType === 'camera') {
      await getMediaStream();
      camElement.srcObject = stream;
      camElement.onloadeddata = await renderCamStream();
      await ui.showProgressComponent('done', 'done', 'done');
      ui.readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
  } catch (error) {
    console.log(error);
    ui.addAlert(error.message);
  }
  ui.handleClick(disabledSelectors, false);
}

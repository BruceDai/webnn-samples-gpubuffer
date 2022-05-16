'use strict';

import {DeepLabV3MNV2Nchw} from './deeplabv3_mnv2_nchw.js';
import {DeepLabV3MNV2Nhwc} from './deeplabv3_mnv2_nhwc.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';
import {Renderer} from './lib/renderer.js';

import { WebGPUBlur } from './blur.js';
import { DeepLabV3MNV2NchwPolyfill } from './deeplabv3_mnv2_nchw_polyfill.js';

const imgElement = document.getElementById('feedElement');
imgElement.src = './images/test.jpg';
const camElement = document.getElementById('feedMediaElement');
const resultElement = document.getElementById('resultMediaElement');
const outputCanvas = document.getElementById('outputCanvas');
let modelName ='deeplabv3mnv2';
let layout = 'nchw';
let instanceType = modelName + layout;
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
let renderer;
let hoverPos = null;
let devicePreference = 'gpu';
let lastDevicePreference = '';
const disabledSelectors = ['#tabs > li', '.btn'];
// An AbortController used to stop the transform.
let abortController = null;
let isPolyfill = false;

$(document).ready(() => {
  $('.icdisplay').hide();
});

$(window).on('load', () => {
  renderer = new Renderer(outputCanvas);
  renderer.setup();
  loadRenderUI();
});

$('#deviceBtns .btn').on('change', async (e) => {
  devicePreference = $(e.target).attr('id');
  if (inputType === 'camera') stopCamera();
  await main();
});

$('#modelBtns .btn').on('change', async (e) => {
  modelName = $(e.target).attr('id');
  if (inputType === 'camera') stopCamera();
  await main();
});

$('#layoutBtns .btn').on('change', async (e) => {
  layout = $(e.target).attr('id');
  if (inputType === 'camera') stopCamera();
  await main();
});

// Click trigger to do inference with <img> element
$('#img').click(async () => {
  if (inputType === 'camera' || stream !== null) stopCamera();
  inputType = 'image';
  $('#pickimage').show();
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
  if (!isFirstTimeLoad) {
    await main();
  }
});

// Click trigger to do inference with <video> media element
$('#cam').click(async () => {
  inputType = 'camera';
  $('#pickimage').hide();
  $('.shoulddisplay').hide();
  await main();
});

function loadRenderUI() {
  const blurSlider = document.getElementById('blurSlider');
  const refineEdgeSlider = document.getElementById('refineEdgeSlider');
  const colorMapAlphaSlider = document.getElementById('colorMapAlphaSlider');
  const selectBackgroundButton = document.getElementById('chooseBackground');
  const clearBackgroundButton = document.getElementById('clearBackground');
  const colorPicker = new iro.ColorPicker('#color-picker-container', {
    width: 200,
    height: 200,
    color: {
      r: renderer.bgColor[0],
      g: renderer.bgColor[1],
      b: renderer.bgColor[2],
    },
    markerRadius: 5,
    sliderMargin: 12,
    sliderHeight: 20,
  });

  $('.bg-value').html(colorPicker.color.hexString);

  colorPicker.on('color:change', (color) => {
    $('.bg-value').html(color.hexString);
    renderer.bgColor = [color.rgb.r, color.rgb.g, color.rgb.b];
  });

  colorMapAlphaSlider.value = renderer.colorMapAlpha * 100;
  $('.color-map-alpha-value').html(renderer.colorMapAlpha);

  colorMapAlphaSlider.oninput = () => {
    const alpha = colorMapAlphaSlider.value / 100;
    $('.color-map-alpha-value').html(alpha);
    renderer.colorMapAlpha = alpha;
  };

  blurSlider.value = renderer.blurRadius;
  $('.blur-radius-value').html(renderer.blurRadius + 'px');

  blurSlider.oninput = () => {
    const blurRadius = parseInt(blurSlider.value);
    $('.blur-radius-value').html(blurRadius + 'px');
    renderer.blurRadius = blurRadius;
  };

  refineEdgeSlider.value = renderer.refineEdgeRadius;

  if (refineEdgeSlider.value === '0') {
    $('.refine-edge-value').html('DISABLED');
  } else {
    $('.refine-edge-value').html(refineEdgeSlider.value + 'px');
  }

  refineEdgeSlider.oninput = () => {
    const refineEdgeRadius = parseInt(refineEdgeSlider.value);
    if (refineEdgeRadius === 0) {
      $('.refine-edge-value').html('DISABLED');
    } else {
      $('.refine-edge-value').html(refineEdgeRadius + 'px');
    }
    renderer.refineEdgeRadius = refineEdgeRadius;
  };

  $('.effects-select .btn input').filter((e) => {
    return e.value === renderer.effect;
  }).parent().toggleClass('active');

  $('.controls').attr('data-select', renderer.effect);

  $('.effects-select .btn').click((e) => {
    e.preventDefault();
    const effect = e.target.children[0].value;
    $('.controls').attr('data-select', effect);
    renderer.effect = effect;
  });

  selectBackgroundButton.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      const img = new Image();
      img.onload = () => {
        renderer.backgroundImageSource = img;
      };
      img.src = URL.createObjectURL(files[0]);
    }
  }, false);

  clearBackgroundButton.addEventListener('click', (e) => {
    renderer.backgroundImageSource = null;
  }, false);

  outputCanvas.addEventListener('mousemove', (e) => {
    const getMousePos = (canvas, evt) => {
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.ceil(evt.clientX - rect.left),
        y: Math.ceil(evt.clientY - rect.top),
      };
    };

    hoverPos = getMousePos(outputCanvas, e);
    renderer.highlightHoverLabel(hoverPos, outputCanvas);
  });

  outputCanvas.addEventListener('mouseleave', (e) => {
    hoverPos = null;
    renderer.highlightHoverLabel(hoverPos, outputCanvas);
  });
}

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
}

async function getMediaStream() {
  // Support 'user' facing mode at present
  const constraints = {audio: false, video: {facingMode: 'user'}};
  stream = await navigator.mediaDevices.getUserMedia(constraints);
}

function stopCamera() {
  camElement.pause();
  camElement.srcObject = null;
  stream.getTracks().forEach((track) => {
    if (track.readyState === 'live' && track.kind === 'video') {
      track.stop();
    }
  });
  abortController.abort();
  abortController = null;
}

let lastComputeTime = 0;
function segmentSemantic() {
  return async (videoFrame, controller) => {
    console.log('- Computing... ');
    const start = performance.now();

    const inputTensor = utils.getInputGPUTensor(videoFrame, inputOptions);
    let outputTensor;
    let segmapTensor;
    let segmapBuffer;
    if (isPolyfill) {
      outputTensor = netInstance.computeOutputGPUTensor(inputTensor);
      segmapTensor = tf.tidy(() => tf.argMax(outputTensor, 1));
      tf.engine().backendInstance.submitQueue();
      segmapBuffer = tf.engine().backendInstance.getBuffer(segmapTensor.dataId);
    } else {
      netInstance.computeGPUTensorToGPUBuffer(inputTensor);
      segmapBuffer = netInstance.getOutputGPUBufferForProcessing();
    }
    await renderer.drawOutput(videoFrame, 
      {buffer: segmapBuffer, width: netInstance.outputWidth, height: netInstance.outputHeight});
    await tf.engine().backendInstance.device.queue.onSubmittedWorkDone();

    const frame_from_canvas = new VideoFrame(outputCanvas, {timestamp: 0});
    controller.enqueue(frame_from_canvas);

    computeTime = (performance.now() - start).toFixed(2);
    console.log(`  done in ${computeTime} ms.`);
    showPerfResult();
    $('#fps').text(`${(1000/computeTime).toFixed(0)} FPS`);
    
    inputTensor.dispose();
    if (isPolyfill) {
      outputTensor.dispose();
      segmapTensor.dispose();
    }
    videoFrame.close();
  };
}

async function renderCamStream() {
  await getMediaStream();
  camElement.srcObject = stream.clone();
  await camElement.play();
  const videoTrack = stream.getVideoTracks()[0];

  const processor = new MediaStreamTrackProcessor({track: videoTrack, maxBufferSize: 256});
  const generator = new MediaStreamTrackGenerator({kind: 'video'});

  const source = processor.readable;
  const sink = generator.writable;

  const transformer = new TransformStream({transform: segmentSemantic()});
  abortController = new AbortController();
  const signal = abortController.signal;

  const popeThroughPromise = source.pipeThrough(transformer, {signal}).pipeTo(sink);

  popeThroughPromise.catch((e) => {
    if (signal.aborted) {
      console.log('Shutting down streams after abort.');
    } else {
      console.error('Error from stream transform:', e);
    }
    source.cancel(e);
    sink.abort(e);
  });

  const processedStream = new MediaStream();
  processedStream.addTrack(generator);
  resultElement.srcObject = processedStream;
  await resultElement.play();
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
    'deeplabv3mnv2nchw': new DeepLabV3MNV2Nchw(),
    'deeplabv3mnv2nchwpolyfill': new DeepLabV3MNV2NchwPolyfill(),
    'deeplabv3mnv2nhwc': new DeepLabV3MNV2Nhwc(),
  };

  return netObject[type];
}

export async function main() {
  try {
    await tf.setBackend('webgpu');
    tf.env().set('WEBGPU_USE_IMPORT', true);
    renderer = new WebGPUBlur(outputCanvas, tf.engine().backendInstance.device);
    ui.handleClick(disabledSelectors, true);
    let start;
    // Set 'numRuns' param to run inference multiple times
    const params = new URLSearchParams(location.search);
    let numRuns = params.get('numRuns');
    numRuns = numRuns === null ? 1 : parseInt(numRuns);
    isPolyfill = params.get('polyfill') === 'true';
    const polyfill = isPolyfill ? 'polyfill' : '';

    if (numRuns < 1) {
      ui.addAlert('The value of param numRuns must be greater than or equal' +
          ' to 1.');
      return;
    }
    // Only do load() and build() when model first time loads,
    // there's new model choosed, and device backend changed
    if (isFirstTimeLoad || instanceType !== modelName + layout + polyfill ||
      lastDevicePreference != devicePreference) {
      if (lastDevicePreference != devicePreference) {
        // Set polyfill backend
        await utils.setPolyfillBackend(devicePreference);
        lastDevicePreference = devicePreference;
      }
      if (netInstance !== null) {
        // Call dispose() to and avoid memory leak
        netInstance.dispose();
      }
      instanceType = modelName + layout + polyfill;
      netInstance = constructNetObject(instanceType);
      inputOptions = netInstance.inputOptions;
      labels = await fetchLabels(inputOptions.labelUrl);
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
      netInstance.build(outputOperand);
      buildTime = (performance.now() - start).toFixed(2);
      console.log(`  done in ${buildTime} ms.`);
    }
    // UI shows inferencing progress
    await ui.showProgressComponent('done', 'done', 'current');
    if (inputType === 'image') {
      $('#outputCanvas').show();
      $('#feedMediaElement').hide();
      $('#resultMediaElement').hide();
      console.log('- Computing... ');
      start = performance.now();
      const inputTensor = utils.getInputGPUTensor(imgElement, inputOptions);
      let outputTensor;
      let segmapTensor;
      let segmapBuffer;
      if (isPolyfill) {
        outputTensor = netInstance.computeOutputGPUTensor(inputTensor);
        segmapTensor = tf.tidy(() => tf.argMax(outputTensor, 1));
        tf.engine().backendInstance.submitQueue();
        segmapBuffer = tf.engine().backendInstance.getBuffer(segmapTensor.dataId);
      } else {
        netInstance.computeGPUTensorToGPUBuffer(inputTensor);
        segmapBuffer = netInstance.getOutputGPUBufferForProcessing();
      }
      await renderer.drawOutput(imgElement, 
        {buffer: segmapBuffer, width: netInstance.outputWidth, height: netInstance.outputHeight});
      await netInstance.device_.queue.onSubmittedWorkDone();
      computeTime = (performance.now() - start).toFixed(2);
      console.log(`  compute time: ${computeTime} ms`);
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').hide();
      ui.readyShowResultComponents();
      showPerfResult();
      inputTensor.dispose();
      if (isPolyfill) {
        outputTensor.dispose();
        segmapTensor.dispose();
      }
    } else if (inputType === 'camera') {
      $('#outputCanvas').hide();
      $('#feedMediaElement').show();
      $('#resultMediaElement').show();
      await renderCamStream();
      await ui.showProgressComponent('done', 'done', 'done');
      $('#fps').show();
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

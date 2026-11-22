import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import mount from '../../streamlit_app/page_actions.js';

// Browser API substitutes: these tests never capture a real screen or microphone.
let cleanup;
let restore;
class Element {
  attributes = {};
  dataset = {};
  handlers = new Map();
  hidden = false;
  disabled = false;
  clicks = 0;
  addEventListener(name, fn) { this.handlers.set(name, fn); }
  removeEventListener(name) { this.handlers.delete(name); }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this[name]; }
  async click() {
    if (this.disabled) return;
    this.clicks += 1;
    await this.handlers.get('click')?.();
  }
}

function setup() {
  const elements = Object.fromEntries([
    '[data-action="print"]', '[data-action="record"]', '.feedback', '.download',
    '[data-tooltip="record"]', '.print-control', '.record-control', '.pdf-actions', '.pdf-control',
  ].map(selector => [selector, new Element()]));
  elements['.download'].hidden = true;
  const track = { stops: 0, onended: null, stop() { this.stops += 1; } };
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
  const state = { prints: 0, picks: 0, blobs: [], revoked: [], recorders: [] };
  const mediaDevices = { async getDisplayMedia(options) {
    state.picks += 1;
    assert.deepEqual(options, {
      video: true, audio: false, preferCurrentTab: true, selfBrowserSurface: 'include',
    });
    return stream;
  } };
  class Recorder {
    state = 'inactive';
    constructor(_stream, options) {
      assert.equal(_stream, stream);
      this.mimeType = options.mimeType;
      state.recorders.push(this);
    }
    static isTypeSupported(type) { return type === 'video/webm'; }
    start() { this.state = 'recording'; }
    stop() {
      this.state = 'inactive';
      queueMicrotask(() => {
        this.ondataavailable({ data: new Blob(['recorded-video'], { type: this.mimeType }) });
        this.onstop();
      });
    }
  }
  const globals = new Map(['window', 'navigator', 'MediaRecorder'].map(
    name => [name, Object.getOwnPropertyDescriptor(globalThis, name)],
  ));
  const create = URL.createObjectURL;
  const revoke = URL.revokeObjectURL;
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {
    print() { state.prints += 1; },
  } });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices } });
  Object.defineProperty(globalThis, 'MediaRecorder', { configurable: true, value: Recorder });
  URL.createObjectURL = blob => { state.blobs.push(blob); return `blob:test-${state.blobs.length}`; };
  URL.revokeObjectURL = url => { state.revoked.push(url); };
  restore = () => {
    for (const [name, descriptor] of globals) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
    URL.createObjectURL = create;
    URL.revokeObjectURL = revoke;
  };
  const parentElement = new Element();
  parentElement.querySelector = selector => elements[selector];
  parentElement.querySelectorAll = () => [
    elements['.pdf-control'], elements['.print-control'], elements['.record-control'],
  ];
  return { elements, track, stream, state, mediaDevices, parentElement };
}

afterEach(async () => {
  cleanup?.();
  await Promise.resolve();
  restore?.();
  cleanup = null;
  restore = null;
});

test('print opens the browser dialog without a capture request', async () => {
  const env = setup();
  cleanup = mount(env);
  await env.elements['[data-action="print"]'].click();
  assert.equal(env.state.prints, 1);
  assert.equal(env.state.picks, 0);
  env.parentElement.handlers.get('keydown')({ key: 'Escape' });
  assert.equal(env.elements['.pdf-control'].dataset.tooltipDismissed, 'true');
  assert.equal(env.elements['.print-control'].dataset.tooltipDismissed, 'true');
  env.elements['.print-control'].handlers.get('pointerenter')();
  assert.equal(env.elements['.print-control'].dataset.tooltipDismissed, undefined);
  env.elements['.record-control'].handlers.get('focusin')();
  assert.equal(env.elements['.record-control'].dataset.tooltipDismissed, undefined);
  env.elements['.pdf-control'].handlers.get('focusin')();
  assert.equal(env.elements['.pdf-control'].dataset.tooltipDismissed, undefined);
});

test('stop creates a video download, keeps its fallback link, and releases capture', async () => {
  const env = setup();
  cleanup = mount(env);
  const button = env.elements['[data-action="record"]'];
  await button.click();
  assert.equal(button.attributes['aria-pressed'], 'true');
  assert.equal(env.elements['[data-tooltip="record"]'].textContent, 'Stop recording');
  assert.equal(env.state.blobs.length, 0);
  await button.click();
  assert.equal(button.attributes['aria-pressed'], 'false');
  assert.equal(env.elements['[data-tooltip="record"]'].textContent, 'Record screen');
  assert.equal(env.track.stops, 1);
  const link = env.elements['.download'];
  assert.equal(link.hidden, false);
  assert.equal(link.clicks, 1);
  assert.match(link.download, /^interview-assistant-.*\.webm$/);
  assert.equal(await env.state.blobs[0].text(), 'recorded-video');
  cleanup();
  cleanup = null;
  assert.deepEqual(env.state.revoked, ['blob:test-1']);
});

test('canceling the sharing picker restores the idle control', async () => {
  const env = setup();
  env.mediaDevices.getDisplayMedia = async () => {
    throw new DOMException('User canceled', 'NotAllowedError');
  };
  cleanup = mount(env);
  const button = env.elements['[data-action="record"]'];
  await button.click();
  assert.equal(button.disabled, false);
  assert.equal(env.elements['[data-tooltip="record"]'].textContent, 'Record screen');
  assert.equal(env.elements['.feedback'].textContent, 'Recording was not started.');
  assert.equal(env.state.blobs.length, 0);
});

test('unmount while the sharing picker is open releases the late stream', async () => {
  const env = setup();
  let resolve;
  env.mediaDevices.getDisplayMedia = () => new Promise(done => { resolve = done; });
  cleanup = mount(env);
  const pending = env.elements['[data-action="record"]'].click();
  cleanup();
  cleanup = null;
  resolve(env.stream);
  await pending;
  assert.equal(env.track.stops, 1);
  assert.equal(env.state.recorders.length, 0);
});

test('recording errors release capture and leave no download', async () => {
  const env = setup();
  cleanup = mount(env);
  const button = env.elements['[data-action="record"]'];
  await button.click();
  env.state.recorders[0].onerror();
  await Promise.resolve();
  assert.equal(env.track.stops, 1);
  assert.equal(button.disabled, false);
  assert.equal(env.elements['[data-tooltip="record"]'].textContent, 'Record screen');
  assert.equal(env.state.blobs.length, 0);
});

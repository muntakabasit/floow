/**
 * Audio Worklet: Processes audio frames on the audio thread
 * Receives audio from mic, sends to main thread via MessagePort
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.port.onmessage = (event) => {
      // Receive config from main thread if needed
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    if (input && input.length > 0) {
      const channelData = input[0];

      // Send audio data to main thread
      this.port.postMessage({
        type: 'audio',
        data: channelData,
        sampleRate: sampleRate,
      });
    }

    // Return true to keep processor alive
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);

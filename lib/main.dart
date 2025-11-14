import 'dart:async';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:fftea/fftea.dart'; // Add to pubspec.yaml: fftea: ^1.0.0
import 'package:fftea/fftea.dart';
import 'dart:typed_data';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Wake Word Detection',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const WakeWordScreen(),
    );
  }
}

class WakeWordScreen extends StatefulWidget {
  const WakeWordScreen({Key? key}) : super(key: key);

  @override
  State<WakeWordScreen> createState() => _WakeWordScreenState();
}

class _WakeWordScreenState extends State<WakeWordScreen> {
  Interpreter? _model;
  final AudioRecorder _audioRecorder = AudioRecorder();
  bool _isListening = false;
  String _detectedWakeWord = '';
  bool _wakeWordDetected = false;
  String _statusMessage = 'Ready to start';
  double _confidence = 0.0;
  final List<double> _audioBuffer = [];

  static const int sampleRate = 16000;
  static const int nMfcc = 40;
  static const int nFft = 400;
  static const int hopLength = 160;
  static const int numMelBins = 40;

  late List<int> _inputShape;
  late List<int> _outputShape;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      setState(() => _statusMessage = 'Loading model...');

      final interpreterOptions = InterpreterOptions()..threads = 2;

      _model = await Interpreter.fromAsset(
        'assets/models/wake_word_model.tflite',
        options: interpreterOptions,
      );

      _inputShape = _model!.getInputTensor(0).shape;
      _outputShape = _model!.getOutputTensor(0).shape;

      print('Input shape: $_inputShape');
      print('Output shape: $_outputShape');

      setState(() => _statusMessage = 'Model loaded successfully');
      _showSnackBar('Model ready! Say "hey barns"');
    } catch (e) {
      setState(() => _statusMessage = 'Error loading model: $e');
      print("Error loading model: $e");
    }
  }

  Future<void> _startListening() async {
    if (_isListening) return;

    var status = await Permission.microphone.request();
    if (!status.isGranted) {
      _showSnackBar('Microphone permission denied');
      return;
    }

    try {
      if (await _audioRecorder.hasPermission()) {
        final stream = await _audioRecorder.startStream(
          const RecordConfig(
            encoder: AudioEncoder.pcm16bits,
            sampleRate: sampleRate,
            numChannels: 1,
          ),
        );

        setState(() {
          _isListening = true;
          _statusMessage = 'Listening for "hey barns"...';
          _wakeWordDetected = false;
        });

        stream.listen(
          (data) => _processAudioData(data),
          onError: (error) {
            _showSnackBar('Error: $error');
            _stopListening();
          },
        );

        _showSnackBar('Started listening');
      }
    } catch (e) {
      _showSnackBar('Error starting audio: $e');
      setState(() => _statusMessage = 'Error: $e');
    }
  }

  void _processAudioData(Uint8List audioData) {
    // Convert bytes to audio samples (Int16)
    for (int i = 0; i < audioData.length - 1; i += 2) {
      int sample = audioData[i] | (audioData[i + 1] << 8);
      if (sample > 32767) sample -= 65536;
      _audioBuffer.add(sample / 32768.0); // Normalize to [-1, 1]
    }

    // Process when we have enough samples (1 second = 16000 samples)
    while (_audioBuffer.length >= sampleRate) {
      final frame = _audioBuffer.sublist(0, sampleRate);
      // Keep 50% overlap for sliding window
      _audioBuffer.removeRange(0, sampleRate ~/ 2);
      // _detectWakeWord(frame);
      _detectWakeWord(List<double>.from(frame));
    }
  }

  List<List<double>> _extractMFCC(List<double> audio) {
    // Step 1: Frame the audio
    int numFrames = ((audio.length - nFft) / hopLength).floor() + 1;
    List<List<double>> frames = [];

    for (int i = 0; i < numFrames; i++) {
      int start = i * hopLength;
      int end = start + nFft;
      if (end > audio.length) break;

      List<double> frame = audio.sublist(start, end);

      // Apply Hamming window
      List<double> windowed = List.generate(frame.length, (j) {
        double window =
            0.54 - 0.46 * math.cos(2 * math.pi * j / (frame.length - 1));
        return frame[j] * window;
      });

      frames.add(windowed);
    }

    // Step 2: Compute power spectrum for each frame
    List<List<double>> powerSpectra = [];
    final fft = FFT(nFft);

    for (var frame in frames) {
      // Pad to FFT length if needed
      while (frame.length < nFft) {
        frame.add(0.0);
      }

      // ✅ FIXED: Use Float64List directly for realFft
      final fftResult = fft.realFft(Float64List.fromList(frame));

      // Compute magnitude (only first half + 1 for real FFT)
      List<double> magnitude = List.generate(nFft ~/ 2 + 1, (i) {
        final re = fftResult[i].x;
        final im = fftResult[i].y;
        return math.sqrt(re * re + im * im);
      });

      powerSpectra.add(magnitude);
    }

    // Step 3: Create Mel filterbank
    List<List<double>> melFilterbank = _createMelFilterbank(
      numMelBins,
      nFft ~/ 2 + 1,
      sampleRate.toDouble(),
    );

    // Step 4: Apply mel filterbank to power spectra
    List<List<double>> melSpectra = [];
    for (var powerSpectrum in powerSpectra) {
      List<double> melSpectrum = List.filled(numMelBins, 0.0);

      for (int i = 0; i < numMelBins; i++) {
        double sum = 0.0;
        for (int j = 0; j < powerSpectrum.length; j++) {
          sum += powerSpectrum[j] * melFilterbank[i][j];
        }
        melSpectrum[i] = sum;
      }

      melSpectra.add(melSpectrum);
    }

    // Step 5: Take log and compute DCT (simplified MFCC)
    List<List<double>> mfccFeatures = [];

    for (var melSpectrum in melSpectra) {
      // Log mel spectrum
      List<double> logMel = melSpectrum.map((x) => math.log(x + 1e-6)).toList();

      // Simple DCT (Type-II)
      List<double> mfcc = List.filled(nMfcc, 0.0);
      for (int k = 0; k < nMfcc; k++) {
        double sum = 0.0;
        for (int n = 0; n < numMelBins; n++) {
          sum += logMel[n] * math.cos(math.pi * k * (n + 0.5) / numMelBins);
        }
        mfcc[k] = sum;
      }

      mfccFeatures.add(mfcc);
    }

    // Step 6: Normalize
    double mean = 0.0;
    double std = 0.0;
    int count = 0;

    for (var frame in mfccFeatures) {
      for (var val in frame) {
        mean += val;
        count++;
      }
    }
    mean /= count;

    for (var frame in mfccFeatures) {
      for (var val in frame) {
        std += (val - mean) * (val - mean);
      }
    }
    std = math.sqrt(std / count);

    // Apply normalization
    for (int i = 0; i < mfccFeatures.length; i++) {
      for (int j = 0; j < mfccFeatures[i].length; j++) {
        mfccFeatures[i][j] = (mfccFeatures[i][j] - mean) / (std + 1e-6);
      }
    }

    return mfccFeatures;
  }

  List<List<double>> _createMelFilterbank(
    int numMelBins,
    int numFreqBins,
    double sampleRate,
  ) {
    // Convert frequency to mel scale
    double hzToMel(double hz) => 2595 * math.log(1 + hz / 700) / math.ln10;
    double melToHz(double mel) => 700 * (math.pow(10, mel / 2595) - 1);

    double lowFreqMel = hzToMel(80.0);
    double highFreqMel = hzToMel(7600.0);

    // Create mel points
    List<double> melPoints = List.generate(
      numMelBins + 2,
      (i) => lowFreqMel + (highFreqMel - lowFreqMel) * i / (numMelBins + 1),
    );

    // Convert back to Hz
    List<double> hzPoints = melPoints.map((m) => melToHz(m)).toList();

    // Convert to FFT bin numbers
    List<int> bins = hzPoints
        .map((hz) => (numFreqBins * hz / (sampleRate / 2)).floor())
        .toList();

    // Create filterbank
    List<List<double>> filterbank = List.generate(
      numMelBins,
      (_) => List.filled(numFreqBins, 0.0),
    );

    for (int i = 1; i <= numMelBins; i++) {
      int leftBin = bins[i - 1];
      int centerBin = bins[i];
      int rightBin = bins[i + 1];

      // Rising slope
      for (int j = leftBin; j < centerBin; j++) {
        if (j < numFreqBins) {
          filterbank[i - 1][j] = (j - leftBin) / (centerBin - leftBin);
        }
      }

      // Falling slope
      for (int j = centerBin; j < rightBin; j++) {
        if (j < numFreqBins) {
          filterbank[i - 1][j] = (rightBin - j) / (rightBin - centerBin);
        }
      }
    }

    return filterbank;
  }

  void _detectWakeWord(List<double> audioFrame) {
    if (_model == null) return;

    try {
      // Extract MFCC features
      List<List<double>> mfccFeatures = _extractMFCC(audioFrame);

      final int timeSteps = mfccFeatures.length;
      if (timeSteps == 0) {
        print('No MFCC features extracted');
        return;
      }

      // Debug: Print shapes to verify
      print('MFCC shape: $timeSteps x $nMfcc');
      print('Expected input shape: $_inputShape');

      // Prepare input tensor matching model's expected shape
      // Model expects: [1, time_steps, n_mfcc, 1]
      var input = List.generate(
        1, // batch size
        (_) => List.generate(
          timeSteps, // time dimension
          (t) => List.generate(
            nMfcc, // feature dimension
            (f) => [mfccFeatures[t][f]], // channel dimension
          ),
        ),
      );

      // Prepare output buffer
      var output = List.generate(
        _outputShape[0],
        (_) => List.filled(_outputShape[1], 0.0),
      );

      // Run inference
      _model!.run(input, output);

      // Extract score
      final score = (output[0][0] is double)
          ? output[0][0] as double
          : double.parse(output[0][0].toString());

      const double threshold = 0.50;

      print('Detection score: ${(score * 100).toStringAsFixed(1)}%');

      if (score > threshold) {
        _onWakeWordDetected('hey barns', score);
      }
    } catch (e, st) {
      print('Detection error: $e');
      print('Stack trace: $st');
    }
  }

  // Keep your existing _extractMFCC and _createMelFilterbank methods unchanged

  void _onWakeWordDetected(String keyword, double confidence) {
    if (!_wakeWordDetected) {
      setState(() {
        _wakeWordDetected = true;
        _detectedWakeWord = keyword;
        _confidence = confidence;
        _statusMessage = 'Wake word detected!';
      });

      print('Wake word detected: $keyword with confidence: $confidence');

      _showSnackBar(
        'Detected: $keyword (${(confidence * 100).toStringAsFixed(1)}%)',
      );

      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('🎉 Wake Word Detected!'),
          content: Text(
            'Wake Word: "$keyword"\n'
            'Confidence: ${(confidence * 100).toStringAsFixed(1)}%\n\n'
            'Action triggered successfully!',
            style: const TextStyle(fontSize: 16),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        ),
      );

      // Reset after 2 seconds
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted && _isListening) {
          setState(() {
            _wakeWordDetected = false;
            _statusMessage = 'Listening for "hey barns"...';
          });
        }
      });
    }
  }

  Future<void> _stopListening() async {
    if (!_isListening) return;

    await _audioRecorder.stop();
    _audioBuffer.clear();

    setState(() {
      _isListening = false;
      _statusMessage = 'Stopped';
      _wakeWordDetected = false;
    });

    _showSnackBar('Stopped listening');
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    _model?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Wake Word Detection'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: _wakeWordDetected
                      ? Colors.green.withOpacity(0.1)
                      : _isListening
                      ? Colors.blue.withOpacity(0.1)
                      : Colors.grey.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  _wakeWordDetected
                      ? Icons.check_circle
                      : _isListening
                      ? Icons.mic
                      : Icons.mic_none,
                  size: 80,
                  color: _wakeWordDetected
                      ? Colors.green
                      : _isListening
                      ? Colors.blue
                      : Colors.grey,
                ),
              ),
              const SizedBox(height: 30),
              Text(
                _statusMessage,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: _wakeWordDetected ? Colors.green : Colors.black87,
                ),
              ),
              const SizedBox(height: 30),
              if (_wakeWordDetected)
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.green[50],
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.green, width: 2),
                  ),
                  child: Column(
                    children: [
                      const Icon(
                        Icons.check_circle,
                        color: Colors.green,
                        size: 48,
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        'Wake Word Detected:',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '"$_detectedWakeWord"',
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.green,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Confidence: ${(_confidence * 100).toStringAsFixed(1)}%',
                        style: TextStyle(fontSize: 14, color: Colors.grey[700]),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 30),
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.blue[200]!),
                ),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.psychology, color: Colors.blue[700]),
                        const SizedBox(width: 8),
                        Text(
                          'Trained Wake Word',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.blue[900],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      '"hey barns"',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              const Spacer(),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isListening ? null : _startListening,
                      icon: const Icon(Icons.play_arrow),
                      label: const Text(
                        'Start',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isListening ? _stopListening : null,
                      icon: const Icon(Icons.stop),
                      label: const Text(
                        'Stop',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.red,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

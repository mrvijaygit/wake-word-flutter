import 'dart:async';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:fftea/fftea.dart';

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

  // CRITICAL FIX: Calculate exact expected frames
  static const int expectedFrames =
      ((sampleRate - nFft) ~/ hopLength) + 1; // = 98

  late List<int> _inputShape;
  late List<int> _outputShape;

  final List<double> _recentScores = [];
  int _consecutiveDetections = 0;

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

      print('✅ Model loaded successfully');
      print('Input shape: $_inputShape');
      print('Output shape: $_outputShape');
      print('Expected frames: $expectedFrames');

      // Verify shape matches
      if (_inputShape.length >= 2 && _inputShape[1] != expectedFrames) {
        print(
          '⚠️  WARNING: Model expects ${_inputShape[1]} frames but code expects $expectedFrames',
        );
      }

      setState(() => _statusMessage = 'Model loaded successfully');
      _showSnackBar('Model ready! Say "hey barns"');
    } catch (e) {
      setState(() => _statusMessage = 'Error loading model: $e');
      print("❌ Error loading model: $e");
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
            print('❌ Stream error: $error');
            _showSnackBar('Error: $error');
            _stopListening();
          },
        );

        _showSnackBar('Started listening');
      }
    } catch (e) {
      print('❌ Error starting audio: $e');
      _showSnackBar('Error starting audio: $e');
      setState(() => _statusMessage = 'Error: $e');
    }
  }

  void _processAudioData(Uint8List audioData) {
    try {
      // Convert bytes to audio samples (Int16)
      for (int i = 0; i < audioData.length - 1; i += 2) {
        int sample = audioData[i] | (audioData[i + 1] << 8);
        if (sample > 32767) sample -= 65536;
        _audioBuffer.add(sample / 32768.0);
      }

      // Process when we have enough samples (1 second)
      while (_audioBuffer.length >= sampleRate) {
        final frame = _audioBuffer.sublist(0, sampleRate);
        _audioBuffer.removeRange(0, sampleRate ~/ 2); // 50% overlap
        _detectWakeWord(List<double>.from(frame));
      }
    } catch (e) {
      print('❌ Audio processing error: $e');
    }
  }

  List<List<double>> _extractMFCC(List<double> audio) {
    try {
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

      // Step 2: Compute power spectrum
      List<List<double>> powerSpectra = [];
      final fft = FFT(nFft);

      for (var frame in frames) {
        final fftResult = fft.realFft(Float64List.fromList(frame));

        List<double> power = List.generate(nFft ~/ 2 + 1, (i) {
          final re = fftResult[i].x;
          final im = fftResult[i].y;
          return (re * re + im * im); // Power (no sqrt)
        });

        powerSpectra.add(power);
      }

      // Step 3: Create Mel filterbank
      List<List<double>> melFilterbank = _createMelFilterbank(
        numMelBins,
        nFft ~/ 2 + 1,
        sampleRate.toDouble(),
      );

      // Step 4: Apply mel filterbank
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

      // Step 5: Log and DCT
      List<List<double>> mfccFeatures = [];

      for (var melSpectrum in melSpectra) {
        List<double> logMel = melSpectrum
            .map((x) => math.log(x + 1e-6))
            .toList();

        // DCT Type-II
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
      int count = 0;

      for (var frame in mfccFeatures) {
        for (var val in frame) {
          mean += val;
          count++;
        }
      }
      mean /= count;

      double std = 0.0;
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
    } catch (e) {
      print('❌ MFCC extraction error: $e');
      return [];
    }
  }

  List<List<double>> _padOrTruncateMfcc(List<List<double>> mfcc) {
    // CRITICAL FIX: Always pad/truncate to EXACT expected frames
    final int target = expectedFrames;

    if (mfcc.isEmpty) {
      return List.generate(target, (_) => List.filled(nMfcc, 0.0));
    }

    if (mfcc.length > target) {
      return mfcc.sublist(0, target);
    } else if (mfcc.length < target) {
      final padding = List<List<double>>.generate(
        target - mfcc.length,
        (_) => List.filled(nMfcc, 0.0),
      );
      return [...mfcc, ...padding];
    } else {
      return mfcc;
    }
  }

  List<List<double>> _createMelFilterbank(
    int numMelBins,
    int numFreqBins,
    double sampleRate,
  ) {
    // HTK mel scale (matches Python)
    double hzToMel(double hz) =>
        2595.0 * math.log(1.0 + hz / 700.0) / math.ln10;
    double melToHz(double mel) => 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0);

    double lowMel = hzToMel(0.0);
    double highMel = hzToMel(sampleRate / 2);

    List<double> melPoints = List.generate(
      numMelBins + 2,
      (i) => lowMel + (highMel - lowMel) * i / (numMelBins + 1),
    );

    List<double> hzPoints = melPoints.map((m) => melToHz(m)).toList();

    List<int> bins = hzPoints
        .map((hz) => ((numFreqBins - 1) * hz / (sampleRate / 2)).floor())
        .toList();

    List<List<double>> filterbank = List.generate(
      numMelBins,
      (_) => List.filled(numFreqBins, 0.0),
    );

    for (int m = 1; m <= numMelBins; m++) {
      int left = bins[m - 1];
      int center = bins[m];
      int right = bins[m + 1];

      // Rising slope
      for (int i = left; i < center; i++) {
        if (i >= 0 && i < numFreqBins) {
          filterbank[m - 1][i] = (i - left) / (center - left);
        }
      }

      // Falling slope
      for (int i = center; i < right; i++) {
        if (i >= 0 && i < numFreqBins) {
          filterbank[m - 1][i] = (right - i) / (right - center);
        }
      }
    }

    return filterbank;
  }

  void _detectWakeWord(List<double> audioFrame) {
    if (_model == null) return;

    try {
      // Extract MFCC
      List<List<double>> mfccFeatures = _extractMFCC(audioFrame);

      if (mfccFeatures.isEmpty) {
        print('⚠️  No MFCC features extracted');
        return;
      }

      // CRITICAL FIX: Ensure exact frame count
      mfccFeatures = _padOrTruncateMfcc(mfccFeatures);

      final int timeSteps = mfccFeatures.length;

      // Verify frame count
      if (timeSteps != expectedFrames) {
        print(
          '❌ Frame count mismatch! Got $timeSteps, expected $expectedFrames',
        );
        return;
      }

      // Prepare input: [1, timeSteps, nMfcc, 1]
      var input = List.generate(
        1,
        (_) => List.generate(
          timeSteps,
          (t) => List.generate(nMfcc, (f) => [mfccFeatures[t][f].toDouble()]),
        ),
      );

      // Prepare output
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

      // IMPROVED: Adaptive thresholding
      const double threshold = 0.70;
      const int requiredConsecutiveDetections = 3;

      // Track scores
      _recentScores.add(score);
      if (_recentScores.length > 10) {
        _recentScores.removeAt(0);
      }

      double avgScore = _recentScores.isNotEmpty
          ? _recentScores.reduce((a, b) => a + b) / _recentScores.length
          : 0.0;

      print(
        'Score: ${(score * 100).toStringAsFixed(1)}% | '
        'Avg: ${(avgScore * 100).toStringAsFixed(1)}%',
      );

      // Detection logic
      if (score > threshold) {
        _consecutiveDetections++;

        if (_consecutiveDetections >= requiredConsecutiveDetections &&
            avgScore > 0.65) {
          _onWakeWordDetected('hey barns', score);
          _consecutiveDetections = 0;
          _recentScores.clear();
        }
      } else {
        if (score < 0.60) {
          _consecutiveDetections = 0;
        }
      }
    } catch (e, stackTrace) {
      print('❌ Detection error: $e');
      print('Stack trace: $stackTrace');
    }
  }

  void _onWakeWordDetected(String keyword, double confidence) {
    if (!_wakeWordDetected) {
      setState(() {
        _wakeWordDetected = true;
        _detectedWakeWord = keyword;
        _confidence = confidence;
        _statusMessage = 'Wake word detected!';
      });

      print(
        '✅ Wake word detected: $keyword (${(confidence * 100).toStringAsFixed(1)}%)',
      );

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

      // Reset after cooldown
      Future.delayed(const Duration(seconds: 3), () {
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
    _recentScores.clear();
    _consecutiveDetections = 0;

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

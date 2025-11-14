import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'openWakeWord - Custom Wake Words',
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
  Interpreter? _heyBarnsModel;
  final AudioRecorder _audioRecorder = AudioRecorder();
  bool _isListening = false;
  String _detectedWakeWord = '';
  bool _wakeWordDetected = false;
  String _statusMessage = 'Ready to start';
  double _confidence = 0.0;
  final List<double> _audioBuffer = [];
  static const int sampleRate = 16000;
  late List<int> _wakeInputShape;
  late TensorType _wakeInputType;
  late List<int> _wakeOutputShape;
  late TensorType _wakeOutputType;
  int _requiredSamples = 0;

  @override
  void initState() {
    super.initState();
    _loadModels();
  }

  Future<void> _loadModels() async {
    try {
      setState(() => _statusMessage = 'Loading models...');

      final interpreterOptions = InterpreterOptions()..threads = 2;

      _heyBarnsModel = await Interpreter.fromAsset(
        'assets/models/wake_word_model.tflite',
        options: interpreterOptions,
      );

      final wakeIn = _heyBarnsModel!.getInputTensor(0);
      final wakeOut = _heyBarnsModel!.getOutputTensor(0);
      _wakeInputShape = wakeIn.shape;
      _wakeInputType = wakeIn.type;
      _wakeOutputShape = wakeOut.shape;
      _wakeOutputType = wakeOut.type;

      print('Wake input shape: $_wakeInputShape, type: $_wakeInputType');
      print('Wake output shape: $_wakeOutputShape, type: $_wakeOutputType');

      // Calculate required samples based on input shape
      // If shape is [1, N] or [1, N, M], calculate total samples needed
      if (_wakeInputShape.length >= 2) {
        _requiredSamples = _wakeInputShape[1];
        if (_wakeInputShape.length > 2) {
          _requiredSamples *= _wakeInputShape[2];
        }
      } else {
        _requiredSamples = 16000; // Default to 1 second if unclear
      }

      print('Required samples for model: $_requiredSamples');

      setState(() => _statusMessage = 'Models loaded successfully');
      _showSnackBar('Models loaded! Ready to detect wake words');
    } catch (e) {
      setState(() => _statusMessage = 'Error loading models: $e');
      print("Error loading model -> $e");
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
          _statusMessage = 'Listening for wake words...';
          _wakeWordDetected = false;
        });

        stream.listen(
          (data) => _processAudioData(data),
          onError: (error) {
            _showSnackBar('Error: $error');
            _stopListening();
          },
        );

        _showSnackBar('Started listening for wake words');
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

    // Process when we have enough samples for the model
    while (_audioBuffer.length >= _requiredSamples) {
      final frame = _audioBuffer.sublist(0, _requiredSamples);
      // Keep overlap for sliding window (50% overlap)
      _audioBuffer.removeRange(0, _requiredSamples ~/ 2);
      _detectWakeWord(frame);
    }
  }

  void _detectWakeWord(List<double> audioFrame) {
    if (_heyBarnsModel == null) return;

    try {
      // Convert to Float32List as required by TFLite
      final Float32List inputBuffer = Float32List.fromList(audioFrame);
      final List output = List.filled(
        _wakeOutputShape.reduce((a, b) => a * b),
        0.0,
      ).reshape([1, _wakeOutputShape[1]]);

      // Reshape input if needed
      final input = inputBuffer.reshape([1, _requiredSamples]);

      _heyBarnsModel!.run(input, output);

      final score = output[0][0];
      const double threshold = 0.5;
      print('Detection score: $score');

      if (score > threshold) {
        _onWakeWordDetected('hey barns', score);
      }
    } catch (e) {
      print('Detection error: $e');
      print('Stack trace: ${StackTrace.current}');
    }
  }

  //second
  // void _detectWakeWord(List<double> audioFrame) {
  //   if (_heyBarnsModel == null) return;

  //   try {
  //     // Convert to Float32List as required by TFLite
  //     final Float32List inputBuffer = Float32List.fromList(audioFrame);
  //     final List output = List.filled(
  //       _wakeOutputShape.reduce((a, b) => a * b),
  //       0.0,
  //     ).reshape([1, _wakeOutputShape[1]]);

  //     // Reshape input if needed
  //     final input = inputBuffer.reshape([1, _requiredSamples]);

  //     _heyBarnsModel!.run(input, output);

  //     final score = output[0][0];
  //     const double threshold = 0.5;

  //     print('Detection score: $score');

  //     if (score > threshold) {
  //       _onWakeWordDetected('hey barns', score);
  //     }
  //   } catch (e) {
  //     print('Detection error: $e');
  //     print('Stack trace: ${StackTrace.current}');
  //   }
  // }

  // void _detectWakeWord(List<double> audioFrame) {
  //   if (_heyBarnsModel == null) {
  //     print('Model not loaded');
  //     return;
  //   }

  //   try {
  //     // Prepare input based on model's expected shape
  //     dynamic input;
  //     dynamic output;

  //     if (_wakeInputShape.length == 2) {
  //       // Shape: [1, N] - flatten audio
  //       input = [Float32List.fromList(audioFrame)];
  //       output = List.generate(
  //         _wakeOutputShape[0],
  //         (i) => Float32List(_wakeOutputShape[1]),
  //       );
  //     } else if (_wakeInputShape.length == 3) {
  //       // Shape: [1, T, F] - reshape audio into time/feature dimensions
  //       int timeSteps = _wakeInputShape[1];
  //       int features = _wakeInputShape[2];

  //       // Create a 2D array of Float32List
  //       List<Float32List> timeStepsList = [];

  //       for (int t = 0; t < timeSteps; t++) {
  //         int start = t * features;
  //         int end = start + features;

  //         if (end > audioFrame.length) {
  //           // Pad with zeros if needed
  //           Float32List featureSlice = Float32List(features);
  //           int available = audioFrame.length - start;
  //           if (available > 0) {
  //             for (int i = 0; i < available; i++) {
  //               featureSlice[i] = audioFrame[start + i];
  //             }
  //           }
  //           // Remaining values are already 0.0 by default
  //           timeStepsList.add(featureSlice);
  //         } else {
  //           Float32List featureSlice = Float32List(features);
  //           for (int i = 0; i < features; i++) {
  //             featureSlice[i] = audioFrame[start + i];
  //           }
  //           timeStepsList.add(featureSlice);
  //         }
  //       }

  //       // Wrap in a list to make it [1, T, F]
  //       input = [timeStepsList];

  //       output = List.generate(
  //         _wakeOutputShape[0],
  //         (i) => Float32List(_wakeOutputShape[1]),
  //       );
  //     } else if (_wakeInputShape.length == 4) {
  //       // Shape: [1, T, F, 1] - 4D tensor
  //       int timeSteps = _wakeInputShape[1];
  //       int features = _wakeInputShape[2];

  //       List<List<Float32List>> timeStepsList = [];

  //       for (int t = 0; t < timeSteps; t++) {
  //         int start = t * features;
  //         int end = start + features;

  //         List<Float32List> featureList = [];

  //         if (end > audioFrame.length) {
  //           // Pad with zeros
  //           for (int f = 0; f < features; f++) {
  //             int idx = start + f;
  //             Float32List singleFeature = Float32List(1);
  //             singleFeature[0] = (idx < audioFrame.length)
  //                 ? audioFrame[idx]
  //                 : 0.0;
  //             featureList.add(singleFeature);
  //           }
  //         } else {
  //           for (int f = 0; f < features; f++) {
  //             Float32List singleFeature = Float32List(1);
  //             singleFeature[0] = audioFrame[start + f];
  //             featureList.add(singleFeature);
  //           }
  //         }

  //         timeStepsList.add(featureList);
  //       }

  //       input = [timeStepsList];

  //       output = List.generate(
  //         _wakeOutputShape[0],
  //         (i) => Float32List(_wakeOutputShape[1]),
  //       );
  //     } else {
  //       print('Unexpected input shape: $_wakeInputShape');
  //       return;
  //     }

  //     _heyBarnsModel!.run(input, output);

  //     // Get prediction score
  //     final score = output[0][0];
  //     const double threshold = 0.5;

  //     print('Detection score: $score');

  //     if (score > threshold) {
  //       _onWakeWordDetected('hey barns', score);
  //     }
  //   } catch (e) {
  //     print('Detection error: $e');
  //     print('Stack trace: ${StackTrace.current}');
  //   }
  // }

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
            _statusMessage = 'Listening for wake words...';
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
    _heyBarnsModel?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('openWakeWord - Free & Offline'),
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
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: Colors.blue[700],
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 12),
                        const Text(
                          '"hey barns"',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
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
              const SizedBox(height: 16),
              Text(
                '100% Free • Offline • No Paid Plans',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[600],
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

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
  Interpreter? _melSpectrogramModel;
  Interpreter? _embeddingModel;
  Interpreter? _heyBarnsModel;

  final AudioRecorder _audioRecorder = AudioRecorder();
  bool _isListening = false;
  String _detectedWakeWord = '';
  bool _wakeWordDetected = false;
  String _statusMessage = 'Ready to start';
  double _confidence = 0.0;

  final List<double> _audioBuffer = [];
  static const int sampleRate = 16000;
  static const int frameSize = 1280; // 80ms at 16kHz
  final int _requiredSamples = 0;

  // Model-specific buffer for 16 frames (1.28 seconds)
  final List<List<double>> _melBuffer = [];
  static const int numFrames = 16;

  late List<int> _melInputShape;
  late TensorType _melInputType;
  late List<int> _melOutputShape;
  late TensorType _melOutputType;

  late List<int> _embeddingInputShape;
  late TensorType _embeddingInputType;
  late List<int> _embeddingOutputShape;
  late TensorType _embeddingOutputType;

  late List<int> _wakeInputShape;
  late TensorType _wakeInputType;
  late List<int> _wakeOutputShape;
  late TensorType _wakeOutputType;

  @override
  void initState() {
    super.initState();
    _loadModels();
  }

  // Updated _loadModels
  Future<void> _loadModels() async {
    try {
      setState(() => _statusMessage = 'Loading models...');

      final interpreterOptions = InterpreterOptions()
        ..threads = 2; // tweak threads

      // _melSpectrogramModel = await Interpreter.fromAsset(
      //   'assets/models/melspectrogram.tflite', // make sure pubspec assets contains this path
      //   options: interpreterOptions,
      // );

      // _embeddingModel = await Interpreter.fromAsset(
      //   'assets/models/embedding_model.tflite',
      //   options: interpreterOptions,
      // );

      _heyBarnsModel = await Interpreter.fromAsset(
        'assets/models/Hey_Barns.tflite',
        options: interpreterOptions,
      );

      // Inspect and store shapes & types:
      // final melIn = _melSpectrogramModel!.getInputTensor(0);
      // final melOut = _melSpectrogramModel!.getOutputTensor(0);
      // _melInputShape = melIn.shape;
      // _melInputType = melIn.type;
      // _melOutputShape = melOut.shape;
      // _melOutputType = melOut.type;
      // print('Mel input shape: $_melInputShape, type: $_melInputType');
      // print('Mel output shape: $_melOutputShape, type: $_melOutputType');

      // final embIn = _embeddingModel!.getInputTensor(0);
      // final embOut = _embeddingModel!.getOutputTensor(0);
      // _embeddingInputShape = embIn.shape;
      // _embeddingInputType = embIn.type;
      // _embeddingOutputShape = embOut.shape;
      // _embeddingOutputType = embOut.type;
      // print(
      //   'Embedding input shape: $_embeddingInputShape, type: $_embeddingInputType',
      // );
      // print(
      //   'Embedding output shape: $_embeddingOutputShape, type: $_embeddingOutputType',
      // );

      final wakeIn = _heyBarnsModel!.getInputTensor(0);
      final wakeOut = _heyBarnsModel!.getOutputTensor(0);
      _wakeInputShape = wakeIn.shape;
      _wakeInputType = wakeIn.type;
      _wakeOutputShape = wakeOut.shape;
      _wakeOutputType = wakeOut.type;
      print('Wake input shape: $_wakeInputShape, type: $_wakeInputType');
      print('Wake output shape: $_wakeOutputShape, type: $_wakeOutputType');

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

    print("Process audio data -> $audioData");

    // Process when we have enough samples (80ms frame)
    while (_audioBuffer.length >= frameSize) {
      final frame = _audioBuffer.sublist(0, frameSize);
      _audioBuffer.removeRange(0, frameSize);
      _detectWakeWord(frame);
    }
  }

  void _detectWakeWord(List<double> audioFrame) {
    if (_heyBarnsModel == null) return;

    try {
      // Convert to Float32List as required by TFLite
      final inputBuffer = Float32List.fromList(audioFrame);

      // Reshape input to match model’s expected input shape [1, N]
      final input = inputBuffer.reshape([1, inputBuffer.length]);

      // Create output buffer as Float32List instead of List<double>
      final outputBuffer = Float32List(
        _wakeOutputShape.reduce((a, b) => a * b),
      );

      // Run inference
      _heyBarnsModel!.run(input, outputBuffer);

      // Access first value (assuming [1,1] output)
      final score = outputBuffer[0];
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

  // void _detectWakeWord(List<double> audioFrame) {
  //   print("detect wake word -> $audioFrame");
  //   if (_melSpectrogramModel == null ||
  //       _embeddingModel == null ||
  //       _heyBarnsModel == null) {
  //     return;
  //   }

  //   try {
  //     // Step 1: Compute mel spectrogramj
  //     // Input shape: [1, 1280] - single audio frame
  //     var inputMel = List.generate(1, (i) => Float32List.fromList(audioFrame));

  //     // Output shape: [1, 32] - mel spectrogram features
  //     var outputMel = List.generate(1, (i) => Float32List(32));

  //     _melSpectrogramModel!.run(inputMel, outputMel);

  //     // Add to buffer (need 16 frames for embedding model)
  //     _melBuffer.add(outputMel[0].toList());
  //     if (_melBuffer.length > numFrames) {
  //       _melBuffer.removeAt(0);
  //     }

  //     // Only run detection when we have enough frames
  //     if (_melBuffer.length == numFrames) {
  //       // Step 2: Get embeddings
  //       // Input shape: [1, 16, 32] - 16 mel frames
  //       var inputEmbedding = List.generate(
  //         1,
  //         (i) => List.generate(
  //           numFrames,
  //           (j) => Float32List.fromList(_melBuffer[j]),
  //         ),
  //       );

  //       // Output shape: [1, 96] - embedding features
  //       var outputEmbedding = List.generate(1, (i) => Float32List(96));

  //       _embeddingModel!.run(inputEmbedding, outputEmbedding);

  //       // Step 3: Run wake word detection
  //       // Input shape: [1, 96] - embedding features
  //       var inputWakeWord = [outputEmbedding[0]];

  //       // Output shape: [1, 1] - detection score
  //       var outputWakeWord = List.generate(1, (i) => Float32List(1));

  //       _heyBarnsModel!.run(inputWakeWord, outputWakeWord);

  //       // Check prediction
  //       final score = outputWakeWord[0][0];
  //       const double threshold = 0.5;

  //       print('Detection score: $score');

  //       if (score > threshold) {
  //         _onWakeWordDetected('hey barns', score);
  //       }
  //     }
  //   } catch (e) {
  //     print('Detection error: $e');
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
    _melBuffer.clear();

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
    _melSpectrogramModel?.close();
    _embeddingModel?.close();
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

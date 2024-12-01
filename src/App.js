import React, { useState, useEffect, useRef } from 'react'; 
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import { Button } from './components/ui/button';
import { Alert, AlertTitle, AlertDescription } from './components/ui/alert';
import { Camera, Users, AlertCircle, Home, BarChart } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const CrowdDetectionApp = () => {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [peopleCount, setPeopleCount] = useState(0);
  const [status, setStatus] = useState('normal');
  const [historicalData, setHistoricalData] = useState([]);
  const [error, setError] = useState(null);

  // Camera detection states
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);

  // Function to send an alert to the backend
  const sendAlertToBackend = async () => {
    try {
      const response = await fetch('http://localhost:5000/send_alert', {  // Update the URL based on your backend
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          cameraName: 'Entrance Camera', // Provide a specific name for your camera
          peopleCount,                  // The current people count
          status,                       // The current crowd density status
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send alert to the backend');
      }

      const result = await response.json();
      console.log('Alert sent successfully:', result);
    } catch (error) {
      console.error('Error sending alert:', error);
    }
  };

  // Update status and historical data based on people count
  useEffect(() => {
    if (peopleCount < 10) {
      setStatus('normal');
    } else if (peopleCount < 20) {
      setStatus('moderate');
    } else {
      setStatus('high');
    }

    // Update historical data
    const newDataPoint = {
      time: new Date().toLocaleTimeString(),
      count: peopleCount,
    };
    setHistoricalData(prev => [...prev.slice(-19), newDataPoint]);

    // Trigger alert if status is high
    if (status === 'high') {
      sendAlertToBackend();
    }
  }, [peopleCount, status]);

  // Initialize camera and detection model
  useEffect(() => {
    if (currentPage === 'camera') {
      let animationFrameId;
      let model;

      const initializeCamera = async () => {
        try {
          model = await window.cocoSsd.load();
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
          });

          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            setIsLoading(false);
          }
        } catch (err) {
          setError('Failed to initialize camera or load model. Please ensure you have granted camera permissions.');
          setIsLoading(false);
        }
      };

      const detectPeople = async () => {
        if (!model || !videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        try {
          const predictions = await model.detect(canvas);
          const peopleDetected = predictions.filter(prediction => prediction.class === 'person');
          setPeopleCount(peopleDetected.length);

          context.strokeStyle = '#00ff00';
          context.lineWidth = 2;
          context.fillStyle = '#00ff00';
          context.font = '16px Arial';

          peopleDetected.forEach(person => {
            const [x, y, width, height] = person.bbox;
            context.strokeRect(x, y, width, height);
            context.fillText('Person: ' + Math.round(person.score * 100) + '%', x, y - 5);
          });

          animationFrameId = requestAnimationFrame(detectPeople);
        } catch (err) {
          setError('Detection error: ' + err.message);
        }
      };

      const loadScripts = async () => {
        try {
          await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
          await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd');
          await initializeCamera();
        } catch (err) {
          setError('Failed to load required libraries: ' + err.message);
        }
      };

      loadScripts();

      if (videoRef.current) {
        videoRef.current.onloadedmetadata = () => {
          detectPeople();
        };
      }

      return () => {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        if (videoRef.current && videoRef.current.srcObject) {
          videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        }
      };
    }
  }, [currentPage]);

  const loadScript = (src) => {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.body.appendChild(script);
    });
  };

  const getStatusInfo = (status) => {
    switch (status) {
      case 'normal': return { color: 'bg-blue-500', message: 'Normal crowd density', description: 'Area is operating within normal capacity' };
      case 'moderate': return { color: 'bg-yellow-500', message: 'Moderate crowding', description: 'Area is getting crowded, monitor situation' };
      case 'high': return { color: 'bg-red-500', message: 'High crowd density', description: 'Area is heavily crowded, action required' };
      default: return { color: 'bg-gray-500', message: 'Status unknown', description: 'Unable to determine crowd density' };
    }
  };

  const statusInfo = getStatusInfo(status);

  const renderDashboard = () => (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center gap-2">
            <Users className="w-6 h-6" />
            Crowd Density Monitor
          </CardTitle>
          <Button onClick={() => setCurrentPage('camera')} className="flex items-center gap-2">
            <Camera className="w-4 h-4" />
            Open Camera
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={`${statusInfo.color} p-6 rounded-lg text-white`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Users className="w-8 h-8" />
              <div>
                <h2 className="text-2xl font-bold">{peopleCount} people detected</h2>
                <p className="text-lg">{statusInfo.message}</p>
              </div>
            </div>
            {status === 'high' && <AlertCircle className="w-8 h-8 animate-pulse" />}
          </div>
        </div>
        <Card className="p-4">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#2563eb" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </CardContent>
    </Card>
  );

  const renderCamera = () => (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center gap-2">
            <Camera className="w-6 h-6" />
            Live Detection
          </CardTitle>
          <Button onClick={() => setCurrentPage('dashboard')} className="flex items-center gap-2">
            <Home className="w-4 h-4" />
            Back to Dashboard
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="text-center p-8">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p>Loading camera and detection model...</p>
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : (
          <>
            <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black">
              <video ref={videoRef} className="absolute inset-0 w-full h-full object-contain" playsInline muted />
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-contain" />
            </div>
            <div className="text-center text-xl font-semibold">
              Currently Detected: {peopleCount} people
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );

  return (
    <div className="w-full max-w-4xl mx-auto p-4 space-y-4">
      {currentPage === 'dashboard' ? renderDashboard() : renderCamera()}
    </div>
  );
};

export default CrowdDetectionApp;

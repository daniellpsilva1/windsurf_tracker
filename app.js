class BarTracker {
    constructor() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.avgVelocityEl = document.getElementById('avgVelocity');
        this.barCoordinatesEl = document.getElementById('barCoordinates');
        this.timePointsEl = document.getElementById('timePoints');
        this.armsCoordinatesEl = document.getElementById('armsCoordinates');
        
        this.startTrackingBtn = document.getElementById('startTracking');
        this.stopTrackingBtn = document.getElementById('stopTracking');
        
        this.tracking = false;
        this.barPositions = [];
        this.leftArmPositions = [];
        this.rightArmPositions = [];
        this.timestamps = [];
        this.model = null;
        this.poseModel = null;
        this.confidenceThreshold = 0.5;
        
        this.setupEventListeners();
        this.loadModels();
    }
    
    async loadModels() {
        try {
            // Load both MobileNet and PoseNet models
            this.model = await tf.loadGraphModel('https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1', { fromTFHub: true });
            this.poseModel = await tf.loadGraphModel('https://tfhub.dev/tensorflow/tfjs-model/movenet/singlepose/lightning/4', { fromTFHub: true });
            console.log('Models loaded successfully');
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }
    
    setupEventListeners() {
        this.startTrackingBtn.addEventListener('click', () => this.startTracking());
        this.stopTrackingBtn.addEventListener('click', () => this.stopTracking());
    }
    
    async startTracking() {
        if (!this.model || !this.poseModel) {
            alert('Please wait for the models to load');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            this.video.srcObject = stream;
            
            this.tracking = true;
            this.barPositions = [];
            this.leftArmPositions = [];
            this.rightArmPositions = [];
            this.timestamps = [];
            
            this.trackBar();
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Could not access camera. Please check permissions.');
        }
    }
    
    async trackBar() {
        if (!this.tracking) return;
        
        // Draw current video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        try {
            // Detect bar and pose
            const [barPosition, poseKeypoints] = await Promise.all([
                this.detectBarWithTensorflow(),
                this.detectPose()
            ]);
            
            if (barPosition) {
                // Draw bar bounding box
                this.ctx.strokeStyle = '#00ff00';
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(
                    barPosition.x,
                    barPosition.y,
                    barPosition.width,
                    barPosition.height
                );

                this.barPositions.push(barPosition);
                this.barCoordinatesEl.textContent = 
                    `Bar: x: ${Math.round(barPosition.x)}, y: ${Math.round(barPosition.y)}`;
            }

            if (poseKeypoints) {
                // Draw arms
                this.drawArms(poseKeypoints);
            }

            this.timestamps.push(Date.now());
            
            if (this.barPositions.length > 1) {
                this.calculateVelocity();
            }
        } catch (error) {
            console.error('Error during detection:', error);
        }
        
        requestAnimationFrame(() => this.trackBar());
    }
    
    async detectBarWithTensorflow() {
        // Convert the canvas image to a tensor
        const tfimg = tf.browser.fromPixels(this.canvas);
        const expandedImg = tfimg.expandDims(0);
        
        // Normalize the image
        const normalizedImg = expandedImg.toFloat().div(255);
        
        // Run object detection
        const predictions = await this.model.executeAsync(normalizedImg);
        
        // Clean up tensors
        tfimg.dispose();
        expandedImg.dispose();
        normalizedImg.dispose();
        
        // Process predictions
        const scores = await predictions[1].data();
        const boxes = await predictions[0].data();
        
        // Find the most likely bar detection
        let maxScore = 0;
        let bestBox = null;
        
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > this.confidenceThreshold && scores[i] > maxScore) {
                const [y, x, height, width] = boxes.slice(i * 4, (i + 1) * 4);
                
                // Convert normalized coordinates to pixel coordinates
                const boxInfo = {
                    x: x * this.canvas.width,
                    y: y * this.canvas.height,
                    width: width * this.canvas.width,
                    height: height * this.canvas.height
                };
                
                maxScore = scores[i];
                bestBox = boxInfo;
            }
        }
        
        // Clean up remaining tensors
        predictions.forEach(t => t.dispose());
        
        return bestBox;
    }
    
    async detectPose() {
        const input = tf.tidy(() => {
            return tf.browser.fromPixels(this.canvas)
                .expandDims(0)
                .div(127.5)
                .sub(1);
        });

        const poses = await this.poseModel.predict(input);
        input.dispose();

        return poses;
    }

    drawArms(poses) {
        const leftShoulder = poses[5];
        const leftElbow = poses[7];
        const leftWrist = poses[9];
        const rightShoulder = poses[6];
        const rightElbow = poses[8];
        const rightWrist = poses[10];

        // Draw left arm
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.beginPath();
        this.ctx.moveTo(leftShoulder.x, leftShoulder.y);
        this.ctx.lineTo(leftElbow.x, leftElbow.y);
        this.ctx.lineTo(leftWrist.x, leftWrist.y);
        this.ctx.stroke();

        // Draw right arm
        this.ctx.strokeStyle = '#0000ff';
        this.ctx.beginPath();
        this.ctx.moveTo(rightShoulder.x, rightShoulder.y);
        this.ctx.lineTo(rightElbow.x, rightElbow.y);
        this.ctx.lineTo(rightWrist.x, rightWrist.y);
        this.ctx.stroke();

        // Update arm coordinates display
        this.armsCoordinatesEl.textContent = 
            `Left Wrist: x: ${Math.round(leftWrist.x)}, y: ${Math.round(leftWrist.y)} | ` +
            `Right Wrist: x: ${Math.round(rightWrist.x)}, y: ${Math.round(rightWrist.y)}`;
    }
    
    calculateVelocity() {
        const positions = this.barPositions;
        const times = this.timestamps;
        
        const lastIndex = positions.length - 1;
        
        // Calculate center points of bounding boxes
        const prevCenter = {
            x: positions[lastIndex - 1].x + positions[lastIndex - 1].width / 2,
            y: positions[lastIndex - 1].y + positions[lastIndex - 1].height / 2
        };
        
        const currentCenter = {
            x: positions[lastIndex].x + positions[lastIndex].width / 2,
            y: positions[lastIndex].y + positions[lastIndex].height / 2
        };
        
        // Calculate distance between centers
        const distance = Math.sqrt(
            Math.pow(currentCenter.x - prevCenter.x, 2) +
            Math.pow(currentCenter.y - prevCenter.y, 2)
        );
        
        const timeDiff = (times[lastIndex] - times[lastIndex - 1]) / 1000;
        const velocity = distance / timeDiff;
        
        // Update UI with smoothed values
        this.avgVelocityEl.textContent = `${velocity.toFixed(2)} px/s`;
        this.timePointsEl.textContent = `${timeDiff.toFixed(3)} seconds`;
    }
    
    stopTracking() {
        this.tracking = false;
        
        const stream = this.video.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        
        if (this.barPositions.length > 1) {
            this.calculateVelocity();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const barTracker = new BarTracker();
});

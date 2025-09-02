// Load environment variables first
require('dotenv').config();

const express = require('express');
const session = require('express-session');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const fetch = require('node-fetch');
const FormData = require('form-data');

const app = express();
const PORT = process.env.PORT || 3000;

// Configure multer for file uploads
const upload = multer({
    dest: 'uploads/',
    limits: {
        fileSize: 50 * 1024 * 1024 // 50MB limit
    }
});

// Session configuration using environment variables
app.use(session({
    secret: process.env.SESSION_SECRET || 'fallback-secret-change-in-production',
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: process.env.COOKIE_SECURE === 'true', // Set to true if using HTTPS
        httpOnly: true,
        maxAge: parseInt(process.env.SESSION_MAX_AGE) || 30 * 60 * 1000 // 30 minutes
    }
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// User credentials from environment variables
const USERS = {
    [process.env.USER_CHAITALI_EMAIL]: {
        password: process.env.USER_CHAITALI_PASSWORD,
        name: process.env.USER_CHAITALI_NAME,
        apiKey: process.env.API_KEY
    },
    [process.env.USER_ANIKETH_EMAIL]: {
        password: process.env.USER_ANIKETH_PASSWORD,
        name: process.env.USER_ANIKETH_NAME,
        apiKey: process.env.API_KEY
    },
    [process.env.USER_ARAVIND_EMAIL]: {
        password: process.env.USER_ARAVIND_PASSWORD,
        name: process.env.USER_ARAVIND_NAME,
        apiKey: process.env.API_KEY
    }
};

// Backend endpoints using environment variables
const BACKEND_HOST = process.env.BACKEND_HOST || 'http://205.147.102.131:8000/';
const BACKEND_ENDPOINTS = {
    '/upload/pdf/proposal_form': `${BACKEND_HOST}/upload/pdf/proposal_form`,
    '/upload/prescription/pdf': `${BACKEND_HOST}/upload/prescription/pdf`,
    '/upload/documents/pdf': `${BACKEND_HOST}/upload/documents/pdf`,
    '/upload/medical/pdf': `${BACKEND_HOST}/upload/medical/pdf`
};

// Middleware to check if user is authenticated
function requireAuth(req, res, next) {
    if (req.session.user) {
        next();
    } else {
        res.status(401).json({ error: 'Authentication required' });
    }
}

// Middleware to validate environment variables
function validateEnvironment() {
    const requiredVars = [
        'SESSION_SECRET',
        'API_KEY',
        'USER_CHAITALI_EMAIL',
        'USER_CHAITALI_PASSWORD',
        'USER_ANIKETH_EMAIL',
        'USER_ANIKETH_PASSWORD',
        'USER_ARAVIND_EMAIL',
        'USER_ARAVIND_PASSWORD'
    ];

    const missingVars = requiredVars.filter(varName => !process.env[varName]);
    
    if (missingVars.length > 0) {
        console.error('Missing required environment variables:', missingVars);
        console.error('Please check your .env file');
        process.exit(1);
    }
}

// Validate environment on startup
validateEnvironment();

// Routes

// Serve login page
app.get('/', (req, res) => {
    if (req.session.user) {
        res.sendFile(path.join(__dirname, 'public', 'IDP_dashboard.html'));
    } else {
        res.sendFile(path.join(__dirname, 'public', 'login.html'));
    }
});

// Serve dashboard (protected)
app.get('/dashboard', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'IDP_dashboard.html'));
});

// Login endpoint
app.post('/login', (req, res) => {
    const { email, password } = req.body;
    
    const user = USERS[email];
    if (user && user.password === password) {
        req.session.user = {
            email: email,
            name: user.name,
            apiKey: user.apiKey
        };
        res.json({ success: true, name: user.name });
    } else {
        res.status(401).json({ error: 'Invalid credentials' });
    }
});

// Logout endpoint
app.post('/logout', (req, res) => {
    req.session.destroy(err => {
        if (err) {
            res.status(500).json({ error: 'Could not log out' });
        } else {
            res.json({ success: true });
        }
    });
});

// Get user info endpoint
app.get('/user', requireAuth, (req, res) => {
    res.json({
        name: req.session.user.name,
        email: req.session.user.email
    });
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV || 'development'
    });
});

// File upload endpoints (protected)
app.post('/upload/pdf/proposal_form', requireAuth, upload.single('file'), handleFileUpload);
app.post('/upload/prescription/pdf', requireAuth, upload.single('file'), handleFileUpload);
app.post('/upload/documents/pdf', requireAuth, upload.single('file'), handleFileUpload);
app.post('/upload/medical/pdf', requireAuth, upload.single('file'), handleFileUpload);
// NEW: Route for our Risk Lens disease prediction
app.post('/api/predict/disease', requireAuth, upload.single('file'), handleDiseasePrediction);

// Handle file upload and forward to backend
async function handleFileUpload(req, res) {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        const endpoint = BACKEND_ENDPOINTS[req.path];
        if (!endpoint) {
            return res.status(400).json({ error: 'Invalid endpoint' });
        }

        console.log(`Forwarding request to: ${endpoint}`);

        // Create form data to send to backend
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path), {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });
        formData.append('api_key', req.session.user.apiKey);

        // Forward request to actual backend
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders(),
            timeout: 300000 // 5 minutes timeout
        });

        // Clean up uploaded file
        fs.unlink(req.file.path, (err) => {
            if (err) console.error('Error deleting temp file:', err);
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Backend error: ${response.status} - ${errorText}`);
            throw new Error(`Backend error: ${response.status} ${response.statusText}`);
        }

        // Check if response is streaming or JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            res.json(data);
        } else {
            // Forward streaming response
            res.setHeader('Content-Type', response.headers.get('content-type') || 'text/plain');
            response.body.pipe(res);
        }

    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: error.message });
    }
}
// NEW (Corrected): Function to handle cataract prediction and call YOUR Python AI server
async function handleDiseasePrediction(req, res) {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        // This is the CORRECT, local address of our Python AI server
        const pythonApiEndpoint = 'http://13.202.6.228:5050/predict'; 

        console.log(`Forwarding Disease prediction request to: ${pythonApiEndpoint}`);

        // Create form data to send to the Python AI server
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path), {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });
        
        // Send the image to the Python AI server
        const response = await fetch(pythonApiEndpoint, {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders(),
            timeout: 120000 // 2 minutes timeout
        });

        // Clean up the temporary uploaded file
        fs.unlink(req.file.path, (err) => {
            if (err) console.error('Error deleting temp file:', err);
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Python AI server error: ${response.status} - ${errorText}`);
            throw new Error(`AI Prediction failed: ${response.statusText}`);
        }

        // Get the JSON result from the Python AI and send it back to the user
        const data = await response.json();
        res.json(data);

    } catch (error) {
        console.error('Cataract prediction error:', error);
        res.status(500).json({ error: "The AI prediction server could not be reached. Please ensure it is running." });
    }
}

// Static assets endpoints (for compatibility)
app.get('/assets/*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', req.path));
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully');
    process.exit(0);
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`Backend host: ${BACKEND_HOST}`);
});

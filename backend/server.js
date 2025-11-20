require('dotenv').config()
const express = require('express')
const connectDB = require('./config/db')
const cors = require('cors')
const fs = require('fs');
const uploadDir = 'uploads/banners/';
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const app = express()

// Middleware
const allowedOrigins = (process.env.CORS_ORIGIN || '').split(',').map(o => o.trim());
app.use(cors({
  origin: function(origin, callback) {
    console.log('CORS check: allowedOrigins =', allowedOrigins, 'incoming origin =', origin);
    // allow requests with no origin (like mobile apps, curl, etc.)
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error('Not allowed by CORS: ' + origin));
  },
  credentials: true
}))
app.use(express.json())

// Connect to MongoDB
connectDB()

// Routes
app.get('/', (req, res) => res.send('API Running'))
app.use('/api/auth', require('./routes/auth'))
app.use('/api/course', require('./routes/course'))
app.use('/uploads', express.static('uploads'))
app.use('/api/user', require('./routes/user'))
app.use('/api/cnn-ai', require('./routes/cnnAi'))
app.use('/api/discussion-forum', require('./routes/discussionForumRoutes/DiscussionForum'))
app.use('/api/submission', require('./routes/submission'))

const PORT = process.env.PORT || 5000
app.listen(PORT, () => console.log(`Server started on port ${PORT}`))


const multer = require('multer');
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError || (err && err.message && err.message.includes('Multer'))) {
    return res.status(400).json({ msg: 'Multer error', error: err.message });
  }
  if (err) {
    return res.status(500).json({ msg: 'Server error', error: err.message });
  }
  next();
});

// models/messageAnalysis.model.js

const mongoose = require("mongoose");

const messageAnalysisSchema = new mongoose.Schema({

    // Input
    messages: {
        type: [String],
        required: true
    },

    is_toxic: Boolean,
    overall_score: Number,
    severity: {
        type: String,
        enum: ["LOW", "MEDIUM", "HIGH"]
    },

    categories: {
        toxic: Number,
        severe_toxic: Number,
        obscene: Number,
        identity_hate: Number,
        insult: Number,
        threat: Number
    },

    detected_language: String,

    ensemble_weights: {
        en_model: Number,
        multi_model: Number
    },

    suggestion: String,
    model_version: String,
    inference_time_ms: Number,

    createdAt: {
        type: Date,
        default: Date.now
    }

}, { timestamps: true });

module.exports = mongoose.model("MessageAnalysis", messageAnalysisSchema);
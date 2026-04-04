// user table or schema

const mongoose = require('mongoose')

const userSchema = new mongoose.Schema({
    phoneNumber: { type: String, unique: true, sparse: true },
    phoneSuffix: { type: String, unique: false },
    username: { type: String },
    email: {
        type: String,
        lowercase: true,
        validate: {
            validator: function (v) {
                return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
            },
            message: "Invalid email address format",
        },
    },
    emailOtp : {type:String},
    emailOtpExpiry : {type:Date},
    profilePicture: {type : String},
    about : {type : String},
    lastSeen : {type: Date},
    isOnline : {type: Boolean, default: false},
    isVarified : {type : Boolean, default : false},
    agreed : {type: Boolean, default : false},
},{timestamps:true});


const User = mongoose.model('User',userSchema);
module.exports = User;
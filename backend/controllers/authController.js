const response = require("../utils/responseHandler");
const User = require("../models/User");
const otpGenrate = require("../utils/otpGenerator");
const sendOtpToEmail = require("../services/emailService");
const { uploadFileToCloudinary } = require('../config/cloudinaryConfig');
const twilloService = require("../services/twilloService");
const generateToken = require("../utils/generateToken");
const Conversation = require("../models/Conversation");


// send-otp 
const sendOtp = async (req, res) => { 
    const { email } = req.body;
    console.log("email",email);
    const otp = otpGenrate();
    const expiry = new Date(Date.now() + 5 * 60 * 1000);
    let user;
    try {
        if (email) {
            user = await User.findOne({ email });

            if (!user) {
                user = new User({ email })
            }
            user.emailOtp = otp;
            user.emailOtpExpiry = expiry;
            await user.save();
            await sendOtpToEmail(email, otp);
            return response(res, 200, 'OTP send to your email', { email })
        }
    }
       /* if (!phoneNumber || !phoneSuffix) {
            return response(res, 400, 'Phone number and Suffix are required')
        }
        const fullPhoneNumber = `${phoneSuffix}${phoneNumber}`;
        user = await User.findOne({ phoneNumber })
        if (!user) {
            user = await new User({ phoneNumber, phoneSuffix, })
        }
        await twilloService.sendOtpToPhoneNumber(fullPhoneNumber);
        await user.save();

        return response(res, 200, 'OTP send successfully', user)*/
     catch (error) {
        console.error(error)
        return response(res, 500, 'internal server error')
    }
}

// step 2 : varify otp ka function

const varifyOtp = async (req, res) => {
    const { email, otp } = req.body;

    try {
        let user;
        if (email) {
            user = await User.findOne({ email });
            if (!user) {
                return response(res, 404, 'User Not Found')
            }

            //  expiry check for otp
            const now = new Date();
            if (!user.emailOtp || String(user.emailOtp) != String(otp) || now > new Date(user.emailOtpExpiry)) {
                return response(res, 400, 'invalid or expired otp')
            }
            user.isVarified = true;
            user.emailOtp = null;
            user.emailOtpExpiry = null;
            await user.save();
        }
        // else phone number se login karega
      /*  else {
            if (!phoneNumber || !phoneSuffix) {
                return response(res, 400, 'Phone number and Suffix are required')
            }
            const fullPhoneNumber = `${phoneSuffix}${phoneNumber}`;
            user = await User.findOne({ phoneNumber });
            if (!user) {
                return response(res, 404, 'User Not Found')
            }
            const result = await twilloService.varifyOtp(fullPhoneNumber, otp);
            if (result.status !== "approved") {
                return response(res, 400, 'Invalid Otp')
            }
            user.isVarified = true;
            await user.save();
        }*/
        const token = generateToken(user?._id); //authentication
        res.cookie("auth_token", token, {
            httpOnly: true,
            maxAge: 1000 * 6024 * 365
        });
        return response(res, 200, 'OTP Varified Successfully', { token, user })
    } catch (error) {
        console.error(error)
        return response(res, 500, 'internal server error')
    }
}


const updateProfile = async (req, res) => {
    const { username, agreed, about } = req.body;
    const userId = req.user.userId;

    try {
        const user = await User.findById(userId);
        const file = req.file;
        if (file) {
            const uploadResult = await uploadFileToCloudinary(file);
            console.log(uploadResult)
            user.profilePicture = uploadResult?.secure_url;
        } else if (req.body.profilePicture) {
            user.profilePicture = req.body.profilePicture;
        }

        if (username) user.username = username;
        if (agreed) user.agreed = agreed;
        if (about) user.about = about;
        await user.save();
        console.log(user);
        return response(res, 200, 'user profile updated successfully', user)
    } catch (error) {
        console.error(error)
        return response(res, 500, "internal server error");
    }
}

const checkAuthenticated = async (req, res) => {
    try {
        const userId = req.user.userId;
        if (!userId) {
            return response(res, 404, 'unauthorized ! please login before access our app')
        }
        const user = await User.findById(userId);
        if (!user) {
            return response(res, '404', 'User not found !');
        }
        return response(res, 200, 'User retrived and allow to use PureChat', user);
    } catch (error) {
        console.error(error);
        return response(res, 500, "Internal Server Error");
    }
}


const getAllUsers = async (req, res) => {
    const loggedInUser = req.user.userId;
    try {
        const users = await User.find({ _id: { $ne: loggedInUser } }).select(  // $ne means -> logged in user ko neglect krke baki saare users
            "username profilePicture lastSeen isOnline about phoneNumber phoneSuffix email"
        ).lean();

        const usersWithConversation = await Promise.all(
            users.map(async (user) => {
                const conversation = await Conversation.findOne({
                    participants: { $all: [loggedInUser, user?._id] }
                }).populate({
                    path: "lastMessage",
                    select: "content createdAt sender receiver"
                }).lean();

                return {
                    ...user,
                    conversation: conversation || null
                }
            })
        );
        return response(res, 200, 'users retrived successfully', usersWithConversation)
    } catch (error) {
        console.error(error);
        return response(res, 500, 'internal server error')
    }
}

const logOut = (req, res) => {
    try {
        res.cookie("auth_token", "", { expires: new Date(0) });
        return response(res, 200, 'logged out successfully...')
    } catch (error) {
        console.error(error);
        return response(res, 500, 'internal server error')
    }
}

module.exports = {
    sendOtp, varifyOtp, updateProfile, checkAuthenticated, getAllUsers, logOut
}
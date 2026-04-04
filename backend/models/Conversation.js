const mongoose = require('mongoose');

const ConversationSchema = new mongoose.Schema({
    participants :[ {type: mongoose.Schema.Types.ObjectId, ref:'User'}],
    lastMessage : {type: mongoose.Schema.Types.ObjectId, ref:'Message'},
    unreadCount : {type:Number, default : 0},
},{timestamps:true})

const conversation = mongoose.model('Conversation',ConversationSchema)

module.exports = conversation;
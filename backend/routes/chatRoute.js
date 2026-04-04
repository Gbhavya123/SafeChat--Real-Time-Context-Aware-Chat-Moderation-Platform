const express = require('express')
const chatController = require('../controllers/chatController');
const authMiddleware = require('../middleware/authMiddleware');
const { multerMiddleware } = require('../config/cloudinaryConfig');
const moderationController = require('../controllers/moderationController')


const router = express.Router();


router.post('/send-message', authMiddleware, multerMiddleware, chatController.sendMessage);
router.get('/conversations', authMiddleware, chatController.getConversation);
router.get('/conversations/:conversationId/messages', authMiddleware, chatController.getMessages)

router.delete('/messages/:messageId', authMiddleware, chatController.deleteMessage)



module.exports = router;

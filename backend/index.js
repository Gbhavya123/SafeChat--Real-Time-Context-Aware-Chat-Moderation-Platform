const { uploadFileToCloudinary } = require('../config/cloudinaryConfig');
const Status = require('../models/Status');
const response = require('../utils/responseHandler');
const Message = require('../models/Message');

exports.createStatus = async (req, res) => {
    try {
        const { content, contentType } = req.body;
        const userId = req.user.userId;
        const file = req.file;

        let mediaUrl = null;
        let finalContentType = contentType || 'text';

        //handle file upload
        if (file) {
            const uploadFile = await uploadFileToCloudinary(file);
            if (!uploadFile?.secure_url) {
                return response(res, 400, 'Failed to upload media');
            }
            mediaUrl = uploadFile?.secure_url;

            if (file.mimetype.startwith("image")) {
                finalContentType = 'image';
            } else if (file.mimetype.startwith('Video')) {
                finalContentType = 'video';
            } else {
                return response(res, 400, 'Unsupported file type');
            }
        } else if (content?.trim()) {
            finalContentType = 'text';
        } else {
            return response(res, 400, "Message content is required");
        }

        const expiresAt = new Date();
        expiresAt.setHours(expiresAt.getHours() + 24) // expiry ko 24 hr tak kar dega

        const status = new Status({
            user: userId,
            content: mediaUrl || content,
            contentType: finalContentType,
            expiresAt,
        });

        await status.save();

        const populatedStatus = await Status.findOne(status?._id)
            .populate("user", "username profilePicture")
            .populate("viewers", "username profilePicture");

        // emit socket event
        if (req.io && req.socketUserMap) {
            // broadcast to all connecting users except the creator
            for (const [connectedUserId, socketId] of req.socketUserMap) {
                if (connectedUserId !== userId) {
                    req.io.to(socketId).emit("new_status", populatedStatus)
                }
            }
        }

        return response(res, 201, "status created successfully", populatedStatus);
    } catch (error) {
        console.error(error);
        return response(res, 500, "Internal server error");
    }
};

exports.getStatuses = async (req, res) => {
    try {
        const statuses = await Status.find({
            expiresAt: { $gt: new Date() },
        })
            .populate("user", "username profilePicture")
            .populate("viewers", "username profilePicture")
            .sort({ createdAt: -1 });

        return response(res, 200, "statuses retrived successfully", statuses)
    } catch (error) {
        console.error(error);
        return response(res, 500, "Internal server error");
    }
}

exports.viewStatus = async (req, res) => {
    const { statusId } = req.params;
    const userId = req.user.userId;
    try {
        const status = await Status.findById(statusId);
        if (!status) {
            return response(res, 404, "Status not found");
        }
        if (!status.viewers.includes(userId)) {
            status.viewers.push(userId);
            await status.save();

            const updateStatus = await Status.findById(statusId)
                .populate("user", " username profilePicture")
                .populate("viewers", "username profilePicture");

            // emit socket event
            if (req.io && req.socketUserMap) {
                // broadcast to all connectiong users excpt the creator
                const statusOwnerSocketId = req.socketUserMap.get(status.user._id.toString())
                if (statusOwnerSocketId) {
                    const viewData = {
                        statusId,
                        viewerId: userId,
                        totalViewers: updateStatus.viewers.length,
                        viewers: updateStatus.viewers
                    }
                    res.io.to(statusOwnerSocketId).emit("status_viewed", viewData)
                } else {
                    console.log('Status owner not connected')
                }
            }
        } else {
            console.log('user already viewed the status')
        }

        return response(res, 200, 'status viewd successfully')
    } catch (error) {
        console.error(error)
        return response(res, 500, 'Internal server error');
    }
}

exports.deleteStatus = async (req, res) => {
    const { statusId } = req.params;
    const userId = req.user.userId;
    try {
        const status = await Status.findById(statusId);
        if (!status) {
            return response(res, 404, "Status not found");
        }
        if (status.user.toString() != userId) {
            return response(res, 403, 'Not authorized to delete the status')
        }
 
        await status.deleteOne();

        // emit socket event 
        if (req.io && req.socketUserMap) {
            for (const [connectedUserId, socketId] of req.socketUserMap) {
                if (connectedUserId !== userId) {
                    req.io.to(socketId).emit("status_deleted", statusId)
                }
            }
        }

        return response(res, 200, 'Status deleted successfully')
    } catch (error) {
        console.error(error);
        return response(res, 500, 'Internal server error');
    }
} 
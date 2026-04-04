import { create } from "zustand";
import { getSocket } from "../services/chat.service";
import axiosInstance from "../services/url.service";
import useUserStore from "./useUserStore";

export const useChatStore = create((set, get) => ({
    conversations: [],
    currentConversation: null,
    currentUser: null,
    messages: [],
    loading: false,
    error: null,
    onlineUsers: new Map(),
    typingUsers: new Map(),

    initsocketListeners: () => {
        const socket = getSocket();
        if (!socket) {
            return;
        }

        socket.off("receive_message");
        socket.off("user_typing");
        socket.off("user_status");
        socket.off("message_send");
        socket.off("message_error");
        socket.off("message_deleted");
        socket.off("messages_read");

        socket.on("receive_message", (message) => {
            get().receiveMessage(message);
        });

        socket.on("message_send", (message) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === message._id ? { ...msg } : msg
                )
            }));
        });

        socket.on("message_status_update", ({ messageId, messageStatus }) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === messageId ? { ...msg, messageStatus } : msg
                )
            }));
        });

        socket.on("messages_read", ({ messageIds }) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    messageIds.includes(msg._id) ? { ...msg, messageStatus: "read" } : msg
                )
            }));
        });

        socket.on("reaction_update", ({ messageId, reactions }) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === messageId ? { ...msg, reactions } : msg
                )
            }));
        });

        socket.on("message_deleted", ({ deletedMessageId }) => {
            set((state) => ({
                messages: state.messages.filter((msg) => msg._id !== deletedMessageId)
            }));
        });

        socket.on("message_error", (error) => {
            console.error("message error", error);
        });

        socket.on("user_typing", ({ userId, conversationId, isTyping }) => {
            set((state) => {
                const newTypingUsers = new Map(state.typingUsers);
                if (!newTypingUsers.has(conversationId)) {
                    newTypingUsers.set(conversationId, new Set());
                }

                const typingSet = newTypingUsers.get(conversationId);

                if (isTyping) {
                    typingSet.add(userId);
                } else {
                    typingSet.delete(userId);
                }

                return { typingUsers: newTypingUsers };
            });
        });

        socket.on("user_status", ({ userId, isOnline, lastSeen }) => {
            set((state) => {
                const newOnlineUsers = new Map(state.onlineUsers);
                newOnlineUsers.set(userId, { isOnline, lastSeen });
                return { onlineUsers: newOnlineUsers };
            });
        });

        const { conversations } = get();
        const currentUser = get().currentUser || useUserStore.getState().user;

        if (conversations?.data?.length > 0 && currentUser?._id) {
            conversations.data.forEach((conv) => {
                const otherUser = conv.participants.find(
                    (participant) => participant._id !== currentUser._id
                );

                if (!otherUser?._id) {
                    return;
                }

                socket.emit("get_user_status", otherUser._id, (status) => {
                    set((state) => {
                        const newOnlineUsers = new Map(state.onlineUsers);
                        newOnlineUsers.set(otherUser._id, status);
                        return { onlineUsers: newOnlineUsers };
                    });
                });
            });
        }
    },

    setCurrentUser: (user) => set({ currentUser: user }),

    fetchConversations: async () => {
        set({ loading: true, error: null });
        try {
            const { data } = await axiosInstance.get("/chats/conversations");
            const currentUser = get().currentUser || useUserStore.getState().user;

            set({ conversations: data, currentUser, loading: false });

            get().initsocketListeners();
            return data;
        } catch (error) {
            set({
                error: error?.response?.data?.message || error?.message,
                loading: false
            });
            return null;
        }
    },

    fetchMessages: async (conversationId) => {
        if (!conversationId) return;

        set({ loading: true, error: null });
        try {
            const { data } = await axiosInstance.get(`/chats/conversations/${conversationId}/messages`);
            const messageArray = data.data || data || [];

            set({
                messages: messageArray,
                currentConversation: conversationId,
                loading: false,
            });

            const { markMessageAsRead } = get();
            markMessageAsRead();

            return messageArray;
        } catch (error) {
            set({
                error: error?.response?.data?.message || error?.message,
                loading: false
            });
            return [];
        }
    },

    sendMessage: async (formData) => {
        const senderId = formData.get("senderId");
        const receiverId = formData.get("receiverId");
        const media = formData.get("media");
        const content = formData.get("content");
        const messageStatus = formData.get("messageStatus");
        const environment = formData.get("environment");
        const moderationTempId = formData.get("tempId");

        const { conversations } = get();
        let conversationId = null;

        if (conversations?.data?.length > 0) {
            const conversation = conversations.data.find(
                (conv) =>
                    conv.participants.some((participant) => participant._id === senderId) &&
                    conv.participants.some((participant) => participant._id === receiverId)
            );

            if (conversation) {
                conversationId = conversation._id;
                set({ currentConversation: conversationId });
            }
        }

        const optimisticId = `temp-${Date.now()}`;
        const optimisticMessage = {
            _id: optimisticId,
            sender: { _id: senderId },
            receiver: { _id: receiverId },
            conversation: conversationId,
            imageOrVideoUrl:
                media && typeof media !== "string"
                    ? URL.createObjectURL(media)
                    : null,
            content,
            contentType: media
                ? media.type.startsWith("image")
                    ? "image"
                    : "video"
                : "text",
            createdAt: new Date().toISOString(),
            messageStatus,
        };

        set((state) => ({
            messages: [...state.messages, optimisticMessage],
        }));

        try {
            const request = moderationTempId
                ? axiosInstance.post("/chats/confirm-send", {
                    senderId,
                    receiverId,
                    finalContent: content,
                    environment,
                    tempId: moderationTempId,
                })
                : axiosInstance.post(
                    "/chats/send-message",
                    formData,
                    {
                        headers: { "Content-Type": "multipart/form-data" },
                    }
                );

            const { data } = await request;
            const messageData = data?.data?.message || data?.data || data;

            if (!messageData?._id) {
                throw new Error(data?.message || "Message was not sent");
            }

            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === optimisticId ? messageData : msg
                ),
            }));

            return messageData;
        } catch (error) {
            console.error("Error sending message", error);

            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === optimisticId
                        ? { ...msg, messageStatus: "failed" } : msg
                ),
                error: error?.response?.data?.message || error?.message,
            }));
            throw error;
        }
    },

    receiveMessage: (message) => {
        if (!message) return;

        const currentUser = get().currentUser || useUserStore.getState().user;
        const { currentConversation, messages, conversations } = get();

        const messageExists = messages.some((msg) => msg._id === message._id);
        if (!messageExists && message.conversation === currentConversation) {
            set((state) => ({
                messages: [...state.messages, message]
            }));

            if (message.receiver?._id === currentUser?._id) {
                get().markMessageAsRead();
            }
        }

        if (conversations?.data) {
            const conversationExists = conversations.data.some(
                (conversation) => conversation._id === message.conversation
            );

            if (conversationExists) {
                set((state) => {
                    const updatedConversations = state.conversations.data.map((conversation) => {
                        if (conversation._id === message.conversation) {
                            return {
                                ...conversation,
                                lastMessage: message,
                                unreadCount: message.receiver?._id === currentUser?._id
                                    ? (currentConversation === message.conversation ? 0 : (conversation.unreadCount || 0) + 1)
                                    : (conversation.unreadCount || 0)
                            };
                        }
                        return conversation;
                    });

                    updatedConversations.sort((a, b) => {
                        const dateA = new Date(a.lastMessage?.createdAt || 0);
                        const dateB = new Date(b.lastMessage?.createdAt || 0);
                        return dateB - dateA;
                    });

                    return {
                        conversations: {
                            ...state.conversations,
                            data: updatedConversations
                        }
                    };
                });
            } else {
                get().fetchConversations();
            }
        }
    },

    markMessageAsRead: async () => {
        const { messages } = get();
        if (!messages.length) return;

        const user = get().currentUser || useUserStore.getState().user;
        if (!user) return;

        const unreadIds = messages
            .filter((msg) => msg.messageStatus !== 'read' && msg.receiver?._id === user._id)
            .map((msg) => msg._id);

        if (unreadIds.length === 0) return;

        try {
            await axiosInstance.put('/chats/messages/read', {
                messageIds: unreadIds
            });

            set((state) => ({
                messages: state.messages.map((msg) =>
                    unreadIds.includes(msg._id) ? { ...msg, messageStatus: "read" } : msg
                )
            }));

            const socket = getSocket();
            if (socket) {
                socket.emit("message_read", {
                    messageIds: unreadIds,
                    senderId: messages[0]?.sender?._id,
                    conversationId: messages[0]?.conversation
                });
            }
        } catch (error) {
            console.error("failed to mark as read", error);
        }
    },

    deleteMessage: async (messageId) => {
        try {
            await axiosInstance.delete(`/chats/messages/${messageId}`);
            set((state) => ({
                messages: state.messages?.filter((msg) => msg?._id !== messageId)
            }));
            return true;

        } catch (error) {
            console.log("error in deleting message", error);
            set({ error: error.response?.data?.message || error.message });
            return false;
        }
    },

    addReaction: async (messageId, emoji) => {
        const socket = getSocket();
        const currentUser = get().currentUser || useUserStore.getState().user;
        if (socket && currentUser) {
            socket.emit("add_reaction", {
                messageId,
                emoji,
                userId: currentUser?._id
            });
        }
    },

    startTyping: (receiverId) => {
        const { currentConversation } = get();
        const socket = getSocket();

        if (socket && currentConversation && receiverId) {
            socket.emit("typing_start", {
                conversationId: currentConversation,
                receiverId
            });
        }
    },

    stopTyping: (receiverId) => {
        const { currentConversation } = get();
        const socket = getSocket();

        if (socket && currentConversation && receiverId) {
            socket.emit("typing_stop", {
                conversationId: currentConversation,
                receiverId
            });
        }
    },

    isUserTyping: (userId) => {
        const { typingUsers, currentConversation } = get();

        if (!currentConversation || !typingUsers.has(currentConversation) || !userId) {
            return false;
        }

        return typingUsers.get(currentConversation).has(userId);
    },

    isUserOnline: (userId) => {
        if (!userId) return null;

        const { onlineUsers } = get();
        return onlineUsers.get(userId)?.isOnline || false;
    },

    getUserLastSeen: (userId) => {
        if (!userId) return null;

        const { onlineUsers } = get();
        return onlineUsers.get(userId)?.lastSeen || null;
    },

    resetMessages: () => set({ messages: [], currentConversation: null }),

    cleanup: () => {
        set({
            conversations: [],
            currentConversation: null,
            currentUser: null,
            messages: [],
            onlineUsers: new Map(),
            typingUsers: new Map(),
        });
    },
}));

// chat ka saara logic isme aayega

import { create } from "zustand";
import { getSocket, initializeSocket } from "../services/chat.service";
import axiosInstance from "../services/url.service";
import useUserStore from "./useUserStore";

export const useChatStore = create((set, get) => ({
    conversations: [], // list of all conversations
    currentConversation: null,
    messages: [],
    loading: false,
    error: null,
    onlineUsers: new Map(),
    typingUsers: new Map(),
 

    // socket event listenrs setup
    initsocketListeners: () => {
        const socket = getSocket();
        if (!socket) {
            return;
        }

        // remove existing events or listeners to prevent duplicate handlers
        socket.off("receive_message");
        socket.off("user_typing");
        socket.off("user_status");
        socket.off("message_send");
        socket.off("message_error");
        socket.off("message_deleted");
        socket.off("messages_read");


        // listen for incoming message
        socket.on("receive_message", (message) => {
            get().receiveMessage(message);
        });

        //confirm message delivery
        socket.on("message_send", (message) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === message._id ? { ...msg } : msg)

            }))
        });
        //update msg status

        socket.on("message_status_update", ({ messageId, messageStatus }) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === messageId ? { ...msg, messageStatus } : msg)
            }))
        });

        // handle bulk messages read (Blue Ticks)
        socket.on("messages_read", ({ messageIds, receiverId }) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    messageIds.includes(msg._id) ? { ...msg, messageStatus: "read" } : msg
                )
            }));
        });

        // handle reaction update
        socket.on("reaction_update", ({ messageId, reactions }) => {
            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === messageId ? { ...msg, reactions } : msg)
            }))
        });

        // handle remove message from local state
        socket.on("message_deleted", ({ deletedMessageId }) => {
            set((state) => ({
                messages: state.messages.filter((msg) => msg._id !== deletedMessageId)
            }))
        });

        // handle any message sending error
        socket.on("message_error", (error) => {
            console.error("message error", error)
        });

        // listener for typing users
        socket.on("user_typing", ({ userId, conversationId, isTyping }) => {
            set((state) => {
                const newTypingUsers = new Map(state.typingUsers);
                if (!newTypingUsers.has(conversationId)) {
                    newTypingUsers.set(conversationId, new Set())
                }
                const typingSet = newTypingUsers.get(conversationId)
                if (isTyping) {
                    typingSet.add(userId)
                }
                else {
                    typingSet.delete(userId)
                }

                return { typingUsers: newTypingUsers }
            })
        });

        // track users online and offline status
        socket.on("user_status", ({ userId, isOnline, lastSeen }) => {
            set((state) => {
                const newOnlineUsers = new Map(state.onlineUsers);
                newOnlineUsers.set(userId, { isOnline, lastSeen });
                return { onlineUsers: newOnlineUsers }
            })
        });

        // emit status check for all users in conversation list
        const { conversations } = get();
        if (conversations?.data?.length > 0) {
            conversations.data?.forEach((conv) => {
                const otherUser = conv.participants.find(
                    (p) => p._id !== get().currentUser._id
                );

                if (otherUser._id) {
                    socket.emit("get_user_status", otherUser._id, (status) => {
                        set((state) => {
                            const newOnlineUsers = new Map(state.onlineUsers);
                            newOnlineUsers.set(state.userId, {
                                isOnline: state.isOnline,
                                lastSeen: state.lastSeen
                            });
                            return { onlineUsers: newOnlineUsers }
                        })
                    })
                }
            })
        }
    },

    setCurrentUser: (user) => set({ currentUser: user }),

    fetchConversations: async () => {
        set({ loading: true, error: null });
        try {
            const { data } = await axiosInstance.get("/chats/conversations");
            set({ conversations: data, loading: false });

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

    // fetch message for a converstion
    fetchMessages: async (conversationId) => {
        if (!conversationId) return;

        set({ loading: true, error: null });
        try {
            const { data } = await axiosInstance.get(`/chats/conversations/${conversationId}/messages`);
            const messageArray = data.data || data || [];

            set(({
                messages: messageArray,
                currentConversation: conversationId,
                loading: false,
            }))

            // mark unread message as read aayega yaha
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

    // send message in real time
    sendMessage: async (formData) => {
        const senderId = formData.get("senderId");
        const receiverId = formData.get("receiverId");
        const media = formData.get("media");
        const content = formData.get("content");
        const messageStatus = formData.get("messageStatus");

        const socket = getSocket();

        const { conversations } = get();
        let conversationId = null;

        if (conversations?.data?.length > 0) {
            const conversation = conversations.data.find(
                (conv) =>
                    conv.participants.some((p) => p._id === senderId) &&
                    conv.participants.some((p) => p._id === receiverId)
            );

            if (conversation) {
                conversationId = conversation._id;
                set({ currentConversation: conversationId });
            }
        }
        // temp message before actual response
        const tempId = `temp-${Date.now()}`;

        const optimisticMessage = {
            _id: tempId,
            sender: { _id: senderId },
            receiver: { _id: receiverId },
            conversation: conversationId,
            imageOrVideoUrl:
                media && typeof media !== "string"
                    ? URL.createObjectURL(media)
                    : null,
            content: content,
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
            const { data } = await axiosInstance.post(
                "/chats/send-message",
                formData,
                {
                    headers: { "Content-Type": "multipart/form-data" },
                }
            );
             
            const messageData = data.data || data;

            // replace optimistic message with real one
            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === tempId ? messageData : msg
                ),
            }));

            return messageData;
        } catch (error) {
            console.error("Error sending message", error);

            set((state) => ({
                messages: state.messages.map((msg) =>
                    msg._id === tempId
                        ? { ...msg, messageStatus: "failed" } : msg),
                error: error?.response?.data?.message || error?.message,
            }));
            throw error;
        }


    },



    receiveMessage: (message) => {
        if (!message) return;

        const { currentConversation, currentUser, messages, conversations } = get();

        // Check if message already exists to prevent duplication
        const messageExists = messages.some((msg) => msg._id === message._id);
        if (!messageExists && message.conversation === currentConversation) {
            set((state) => ({
                messages: [...state.messages, message]
            }));

            // automatically mark as read
            if (message.receiver?._id === currentUser?._id) {
                get().markMessageAsRead();
            }
        }

        // update conversation preview and unread count
        if (conversations?.data) {
            const conversationExists = conversations.data.some(c => c._id === message.conversation);

            if (conversationExists) {
                set((state) => {
                    const updatedConversations = state.conversations.data.map((conv) => {
                        if (conv._id === message.conversation) {
                            return {
                                ...conv,
                                lastMessage: message,
                                unreadCount: message.receiver?._id === currentUser?._id
                                    ? (currentConversation === message.conversation ? 0 : (conv.unreadCount || 0) + 1)
                                    : (conv.unreadCount || 0)
                            };
                        }
                        return conv;
                    });

                    // sorting: move updated conversation to top
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
                // Ideally trigger a refetch if conversation is new and doesn't exist in list yet
                // Or construct a new conversation object if we had enough info, but fetching is safer
                get().fetchConversations();
            }
        }
    },

    // mark as read function
    markMessageAsRead: async () => {
        const { messages, HZcurrentUser } = get();
        if (!messages.length || !HZcurrentUser) {
            // Try to get user if HZcurrentUser is undefined, though it should be set
            const user = useUserStore.getState().user;
            if (!user) return;
        }

        // Use currentUser from store or fallback
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
                    unreadIds.includes(msg._id) ? { ...msg, messageStatus: "read" } : msg)
            }));

            const socket = getSocket();
            if (socket) {
                socket.emit("message_read", {
                    messageIds: unreadIds,
                    senderId: messages[0]?.sender?._id, // Assuming conversation is with one person
                    conversationId: messages[0]?.conversation
                })
            }
        } catch (error) {
            console.error("failed to mark as read", error);
        }
    },


    // delete message
    deleteMessage: async (messageId) => {
        try {
            await axiosInstance.delete(`/chats/messages/${messageId}`);
            set((state) => ({
                messages: state.messages?.filter((msg) => msg?._id !== messageId)
            }))
            return true;

        } catch (error) {
            console.log("error in deleting message", error);
            set({ error: error.response?.data?.message || error.message })
            return false;
        }
    },



    // add or change reactions
    addReaction: async (messageId, emoji) => {
        const socket = getSocket();
        const { currentUser } = get();
        if (socket && currentUser) {
            socket.emit("add_reaction", {
                messageId, emoji,
                userId: currentUser?._id
            })
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
            messages: [],
            onlineUsers: new Map(),
            typingUsers: new Map(),
        });
    },
}));
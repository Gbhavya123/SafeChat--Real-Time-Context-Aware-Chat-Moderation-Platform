import { io } from 'socket.io-client'
import useUserStore from '../store/useUserStore'


let socket = null;

export const initializeSocket = () => {
    if (socket) return socket;

    const user = useUserStore.getState().user;

    const BACKEND_URL = process.env.REACT_APP_API_URL;

    socket = io(BACKEND_URL, {
        withCredentials: true,
    });

    //connection events (related to backend socket)

    socket.on("connect", () => {
        const user = useUserStore.getState().user;
        console.log("🟢 Socket CONNECTED:", socket?.id, "User:", user?._id);
        if (user) {
            socket.emit("user_connected", user._id);
            console.log("📤 Emitted user_connected:", user._id);
        } else {
            console.warn("⚠️ Socket connected but NO USER found in store!");
        }
    })

    socket.on("connect_error", (error) => {
        console.error("🔴 Socket Connection ERROR:", error);
    }) 

    socket.on("disconnect", (reason) => {
        console.warn("🔸 Socket DISCONNECTED:", reason);
    })

    // Fallback: if socket is already connected (rare race condition or reused socket)
    if (socket.connected) {
        const user = useUserStore.getState().user;
        console.log("🟢 Socket ALREADY CONNECTED:", socket.id);
        if (user) {
            socket.emit("user_connected", user._id);
            console.log("📤 Emitted user_connected (Immediate):", user._id);
        }
    }

    return socket;
}

export const getSocket = () => {
    if (!socket) {
        return initializeSocket();
    }
    return socket;
}

export const disconnectSocket = () => {
    if (socket && socket.connected) {
        socket.disconnect();
    }
    socket = null;
};
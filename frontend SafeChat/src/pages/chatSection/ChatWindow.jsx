import React, { useRef, useState, useEffect } from "react";
import { format } from "date-fns";
import { FaArrowLeft, FaTimes, FaToggleOn, FaToggleOff, FaPaperPlane } from "react-icons/fa";

import useThemeStore from "../../store/themeStore";
import useUserStore from "../../store/useUserStore";
import { useChatStore } from "../../store/chatStore";
import emptyWindowImage from "../../images/_image_use.png";
import MessageBubble from "./MessageBubble";
import { analyzeToxicity } from "../../services/toxicityApi";

const ChatWindow = ({ selectedContact, setSelectedContact }) => {
  const { theme } = useThemeStore();
  const { user } = useUserStore();
  const {
    conversations,
    fetchMessages,
    messages,
    sendMessage,
    isUserTyping,
    isUserOnline,
    resetMessages,
  } = useChatStore();

  const [toggleMode, setChatMode] = useState("non-professional");
  const [text, setText] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const messageEndRef = useRef(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (selectedContact?._id) {
      const conversationData = conversations?.data || [];
      const conversation = conversationData.find((item) =>
        item.participants.some((participant) => participant._id === selectedContact._id)
      );

      if (conversation) {
        fetchMessages(conversation._id);
      } else {
        resetMessages();
      }
    }
  }, [selectedContact, conversations, fetchMessages, resetMessages]);

  const toggleChatMode = () => {
    setChatMode((prev) => prev === "professional" ? "non-professional" : "professional");
    setAnalysisResult(null);
  };

  const resolveEnvironment = () =>
    toggleMode === "professional" ? "OFFICE" : "GAMING";

  const handleSendAction = async () => {
    if (!text.trim()) return;

    try {
      const currentText = text.trim();
      const data = await analyzeToxicity({
        message: currentText,
        senderId: user?._id,
        receiverId: selectedContact?._id,
        environment: resolveEnvironment(),
      });

      if (data.action === "ALLOW") {
        await handleFinalSend(currentText, data.tempId);
        return;
      }

      setAnalysisResult({
        ...data,
        originalContent: currentText,
      });
    } catch (error) {
      console.error("Toxicity analysis failed:", error);
    }
  };

  const handleConfirmSend = async (contentToSend) => {
    try {
      await handleFinalSend(contentToSend, analysisResult?.tempId || null);
    } catch (error) {
      console.error("Confirm send failed:", error);
    }
  };

  const handleFinalSend = async (messageContent, tempId = null) => {
    if (!selectedContact || !messageContent.trim()) return;

    try {
      const formData = new FormData();
      formData.append("senderId", user?._id);
      formData.append("receiverId", selectedContact?._id);
      formData.append("environment", resolveEnvironment());
      formData.append("content", messageContent.trim());
      formData.append("messageStatus", online ? "delivered" : "sent");

      if (tempId) {
        formData.append("tempId", tempId);
      }

      if (selectedFile) {
        formData.append("media", selectedFile, selectedFile.name);
      }

      await sendMessage(formData);
      setText("");
      setAnalysisResult(null);
      setSelectedFile(null);
    } catch (error) {
      console.error("failed to send message", error);
      throw error;
    }
  };

  const online = isUserOnline(selectedContact?._id);
  const isTyping = isUserTyping(selectedContact?._id);

  const groupedMessages = Array.isArray(messages)
    ? messages.reduce((acc, message) => {
        if (!message.createdAt) return acc;
        const dateString = format(new Date(message.createdAt), "yyyy-MM-dd");
        if (!acc[dateString]) acc[dateString] = [];
        acc[dateString].push(message);
        return acc;
      }, {})
    : {};

  if (!selectedContact) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center h-screen text-center p-4">
        <img src={emptyWindowImage} alt="chat-app" className="w-64 mb-4" />
        <h2 className={`text-2xl font-semibold ${theme === "dark" ? "text-white" : "text-black"}`}>
          Select a contact to start
        </h2>
      </div>
    );
  }

  return (
    <div className="flex-1 h-screen w-full flex flex-col overflow-hidden">
      <div className={`p-3 md:p-4 flex items-center justify-between border-b ${theme === "dark" ? "bg-[#303430] text-white border-gray-700" : "bg-zinc-100 text-gray-800 border-gray-200"}`}>
        <div className="flex items-center min-w-0">
          <button onClick={() => setSelectedContact(null)} className="mr-2 md:mr-3 hover:opacity-70 shrink-0">
            <FaArrowLeft size={18} />
          </button>
          <img src={selectedContact?.profilePicture} className="w-8 h-8 md:w-10 md:h-10 rounded-full shrink-0 object-cover" alt="profile" />
          <div className="ml-2 md:ml-3 truncate">
            <h2 className="font-semibold text-sm md:text-base truncate">{selectedContact?.username}</h2>
            <p className="text-[10px] md:text-xs opacity-70">
              {isTyping ? "Typing..." : online ? "Online" : "Offline"}
            </p>
          </div>
        </div>

        <div
          onClick={toggleChatMode}
          className={`flex items-center cursor-pointer px-2 py-1 md:px-3 md:py-1 rounded-full border transition-all duration-300 shrink-0 ml-2 ${
            toggleMode === "professional"
              ? "bg-purple-500/10 border-purple-500 text-purple-500"
              : "bg-gray-500/10 border-gray-400 text-gray-400"
          }`}
        >
          <span className="hidden sm:inline text-[10px] font-bold uppercase mr-2 tracking-wider">
            {toggleMode === "professional" ? "Professional" : "Casual"}
          </span>
          <span className="sm:hidden text-[10px] font-bold uppercase mr-1.5 tracking-wider">
            {toggleMode === "professional" ? "Pro" : "Cas"}
          </span>
          {toggleMode === "professional" ? <FaToggleOn size={18} /> : <FaToggleOff size={18} />}
        </div>
      </div>

      <div className={`flex-1 p-4 overflow-y-auto ${theme === "dark" ? "bg-[#191a1a]" : "bg-[#f1ece5]"}`}>
        {Object.entries(groupedMessages).map(([date, msgs]) => (
          <React.Fragment key={date}>
            <div className="flex justify-center my-4">
              <span className="text-xs px-3 py-1 bg-gray-500/20 rounded-full">{date}</span>
            </div>
            {msgs.map((msg) => (
              <MessageBubble key={msg._id} message={msg} theme={theme} currentUser={user} />
            ))}
          </React.Fragment>
        ))}
        <div ref={messageEndRef} />
      </div>

      <div className={`p-4 border-t ${theme === "dark" ? "bg-[#303430] border-gray-700" : "bg-white border-gray-200"}`}>
        {analysisResult && (
          <div className={`mb-3 p-4 rounded-lg border ${
            analysisResult.action === "BLOCK"
              ? "bg-red-50 dark:bg-red-900/20 border-red-500 text-red-700 dark:text-red-400"
              : analysisResult.action === "SUGGEST"
                ? (analysisResult.severity?.toLowerCase() === "high"
                    ? "bg-orange-50 dark:bg-orange-900/20 border-orange-500 text-orange-700 dark:text-orange-400"
                    : "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500 text-yellow-700 dark:text-yellow-400")
                : theme === "dark" ? "bg-gray-800 text-white border-gray-600" : "bg-gray-50 text-black border-gray-300"
          }`}>
            <div className="flex justify-between items-start mb-2">
              <span className="font-semibold text-sm">
                {analysisResult.action === "BLOCK" && "Write a new better message"}
                {analysisResult.action === "SUGGEST" && "Suggestion available"}
              </span>
              <button onClick={() => setAnalysisResult(null)} className="opacity-70 hover:opacity-100 transition">
                <FaTimes size={14} />
              </button>
            </div>

            {analysisResult.suggestion && (
              <div className="mb-4 text-sm">
                <span className="block mb-1 opacity-80 text-xs uppercase tracking-wider font-semibold">
                  Suggested Alternative:
                </span>
                <p className="italic bg-white/40 dark:bg-black/20 p-2 rounded border border-white/20 mb-2">
                  "{analysisResult.suggestion}"
                </p>

                {analysisResult.action === "BLOCK" && (
                  <div className="flex justify-end">
                    <button
                      onClick={() => handleConfirmSend(analysisResult.suggestion)}
                      className="py-1 px-4 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded shadow transition duration-200"
                    >
                      Use Suggestion
                    </button>
                  </div>
                )}
              </div>
            )}

            {analysisResult.action === "SUGGEST" && (
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => handleConfirmSend(analysisResult.suggestion)}
                  className="flex-1 py-1.5 px-3 bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white text-sm font-semibold rounded-md shadow transition duration-200"
                >
                  Use Suggestion
                </button>
                <button
                  onClick={() => handleConfirmSend(analysisResult.originalContent)}
                  className="flex-1 py-1.5 px-3 bg-gray-500 hover:bg-gray-600 text-white text-sm font-semibold rounded-md shadow transition duration-200"
                >
                  Send as it is
                </button>
              </div>
            )}
          </div>
        )}

        <div className="flex items-end gap-2">
          <textarea
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              if (analysisResult) setAnalysisResult(null);
            }}
            className={`flex-1 p-3 rounded-xl transition-all resize-none min-h-[50px] max-h-[150px] focus:outline-none focus:ring-2 focus:ring-purple-500/50 ${theme === "dark" ? "bg-[#191a1a] text-white border border-gray-700" : "bg-gray-100 text-black border border-gray-200"}`}
            placeholder={`Type a ${toggleMode === "professional" ? "professional" : "casual"} message...`}
            rows={1}
            style={{ height: text ? "auto" : "50px" }}
          />

          <button
            onClick={handleSendAction}
            disabled={!text.trim() || analysisResult !== null}
            className={`mb-1 p-3.5 rounded-full flex items-center justify-center transition-all duration-300 shrink-0 ${
              !text.trim() || analysisResult !== null
                ? "bg-gray-400 dark:bg-gray-700 text-white cursor-not-allowed opacity-50"
                : "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md hover:shadow-lg transform active:scale-95"
            }`}
          >
            <FaPaperPlane size={18} className="translate-y-[1px] translate-x-[-1px]" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatWindow;

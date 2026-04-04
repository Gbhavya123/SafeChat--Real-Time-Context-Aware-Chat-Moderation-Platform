import { format } from "date-fns";
import React, { useRef, useState } from "react";
import { FaCheck, FaCheckDouble, FaSmile } from "react-icons/fa";
import { HiDotsVertical } from 'react-icons/hi'
import useOutsideClick from "../../hooks/useOutsideclick";



const MessageBubble = ({
    message,
    theme,
    onReact,
    currentUser,
    deleteMessage,
}) => {
    const [showReactions, setShowReactions] = useState(false);
    const [showOptions, setShowOptions] = useState(false);

    const messageRef = useRef(null);
    const optionsRef = useRef(null);

    useOutsideClick(optionsRef, () => setShowOptions(false));


    const isUserMessage = message.sender?._id === currentUser?._id;

    // Use standard Flexbox alignment
    const containerClass = `flex w-full mb-1 ${isUserMessage ? 'justify-end' : 'justify-start'}`;


    const bubbleBg = isUserMessage
        ? (theme === "dark" ? "bg-[#005c4b] text-white" : "bg-[#d9fdd3] text-black")
        : (theme === "dark" ? "bg-[#202c33] text-white" : "bg-white text-black");

    const bubbleReview = `relative max-w-[65%] rounded-lg px-2 py-1 shadow-sm text-sm group ${bubbleBg} ${isUserMessage ? 'rounded-tr-none' : 'rounded-tl-none'}`;

    const handleReact = (emoji) => {
        onReact(message._id, emoji);
        setShowReactions(false);
    };





    if (!message) return null;

    return (
        <div className={containerClass}>
            <div className={bubbleReview} ref={messageRef}>
                {/* Message Content */}
                <div className="flex flex-col relative z-10">
                    <div className="mr-6 mb-1 break-words whitespace-pre-wrap"> {/* mr-6 reserves space for the dropdown arrow */}
                        {message.contentType === "text" && (
                            <span className="leading-relaxed">{message.content}</span>
                        )}

                        {message.contentType === "image" && (
                            <div className="-mx-1 -mt-1 sm:-mx-2 sm:-mt-2 mb-1">
                                <img
                                    src={message.imageOrVideoUrl}
                                    alt="attachment"
                                    className="rounded-lg w-full h-auto object-cover max-h-[300px]"
                                />
                                {message.content && <p className="px-1 mt-1">{message.content}</p>}
                            </div>
                        )}
                    </div>

                    {/* Meta Row: Time & status */}
                    <div className="flex items-center justify-end gap-1 text-[11px] opacity-70 leading-none pb-1 self-end min-w-[50px] ml-auto -mt-2">
                        <span>
                            {message.createdAt ? format(new Date(message.createdAt), "HH:mm") : ""}
                        </span>
                        {isUserMessage && (
                            <span className="ml-[2px] flex items-center">
                                {message.messageStatus === "read" ? (
                                    <FaCheckDouble className="text-blue-500" size={14} />
                                ) : message.messageStatus === "delivered" ? (
                                    <FaCheckDouble size={14} />
                                ) : (
                                    <FaCheck size={12} />
                                )}
                            </span>
                        )}
                    </div>
                </div>

                {/* Hover Options (Three dots) - Positioned Top Right */}
                <button
                    onClick={() => setShowOptions((prev) => !prev)}
                    className={`absolute top-1 right-1 p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity z-20 ${theme === "dark" ? "bg-black/20 hover:bg-black/30" : "bg-gray-200/50 hover:bg-gray-300"}`}
                >
                    <HiDotsVertical size={14} className={theme === "dark" ? "text-gray-300" : "text-gray-600"} />
                </button>

                {/* Hover Reaction (Smile) - Positioned Outside */}
                {/* <button
                    onClick={() => setShowReactions(!showReactions)}
                    className={`absolute top-1 ${isUserMessage ? '-left-8' : '-right-8'} p-1.5 rounded-full opacity-0 group-hover:opacity-100 transition-all z-10 ${theme === "dark" ? "bg-[#202c33] text-gray-300" : "bg-white text-gray-500"} shadow-sm border border-gray-100 dark:border-gray-700`}
                >
                    <FaSmile className={`${theme === 'dark'?"text-gray-300":"text-gray-600"}`} />
                </button> */}

                {/* Reaction Picker Popover */}
                {/* {showReactions && (
                    <div className={`absolute -top-10 ${isUserMessage ? 'right-0' : 'left-0'} flex gap-1 bg-white dark:bg-[#202c33] p-1 rounded-full shadow-lg border dark:border-gray-700 z-50`}>
                        {["👍", "❤️", "😂", "😮", "😢", "🙏"].map((emoji) => (
                            <button
                                key={emoji}
                                onClick={() => handleReact(emoji)}
                                className="hover:scale-125 transition-transform text-lg px-1"
                            >
                                {emoji}
                            </button>
                        ))}
                    </div>
                )} */}

                {/* Options Menu (Delete etc) */}
                {/* Options Menu (Delete etc) */}
                {showOptions && (
                    <div ref={optionsRef} className={`absolute right-4 top-6 w-32 ${theme === "dark" ? "bg-[#233138] text-gray-200" : "bg-white text-gray-800"} shadow-xl rounded-md py-2 text-sm z-50 text-left flex flex-col`}>
                        <button
                            className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                            onClick={() => {
                                navigator.clipboard.writeText(message.content || "");
                                setShowOptions(false);
                            }}
                        >
                            Copy
                        </button>

                        {isUserMessage && (
                            <button
                                className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-red-500"
                                onClick={() => {
                                    deleteMessage(message._id);
                                    setShowOptions(false);
                                }}
                            >
                                Delete
                            </button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default MessageBubble;
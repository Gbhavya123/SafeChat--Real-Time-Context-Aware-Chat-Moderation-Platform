import axiosInstance from "./url.service";

export const analyzeToxicity = async ({
  message,
  senderId,
  receiverId,
  environment,
}) => {
  const response = await axiosInstance.post("/chats/analyze-message", {
    message,
    senderId,
    receiverId,
    environment,
  });

  return response.data;
};

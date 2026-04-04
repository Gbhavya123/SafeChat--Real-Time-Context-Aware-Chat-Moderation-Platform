import { create } from "zustand";
import { persist } from "zustand/middleware";

const useLayoutStore = create(
  persist(
    (set) => ({
      activeTab: "chats",
      selectedContact: null,
      setSelectedContact: (contact) => set({ selectedContact: contact }),
      setActiveTab: (tab) => set({ activeTab: tab }),

      // Read Receipts
      readReceipts: true,
      setReadReceipts: (value) => set({ readReceipts: value }),
    }),
    {
      name: "layout-storage",
      getStorage: () => localStorage,
    }
  )
);

export default useLayoutStore;

import { defineStore } from "pinia";
import { computed, ref } from "vue";

type MessageListener = (data: string) => void;

export const useWebsocketStore = defineStore("websockets", () => {
    const socket = ref<WebSocket | undefined>(undefined);
    const status = ref<"connecting" | "open" | "closed">("closed");
    const url = computed(() => socket.value?.url);
    const errors = ref<Event[]>([]);
    const listeners = ref<Set<MessageListener>>(new Set());

    const subscribe = (listener: MessageListener) => {
        listeners.value.add(listener);
        return () => listeners.value.delete(listener);
    };

    const disconnect = () => {
        if (socket.value) socket.value.close();
    };
    const connect = (
        url: string,
        options?: {
            onOpen?: () => void;
            onMessage?: MessageListener;
            onClose?: () => void;
            onError?: (error: Event) => void;
        }
    ) => {
        const websocket = new WebSocket(url);

        const dispatch = (decoded: string) => {
            options?.onMessage?.(decoded);
            for (const listener of listeners.value) listener(decoded);
        };

        websocket.onopen = () => {
            status.value = "open";
            options?.onOpen?.();
        };
        websocket.onmessage = event => {
            const data = event.data;

            if (typeof data === "string") {
                dispatch(data);
            } else if (data instanceof ArrayBuffer) {
                const decoder = new TextDecoder();
                dispatch(decoder.decode(data));
            } else if (data instanceof Blob) {
                const reader = new FileReader();
                reader.onload = () => dispatch(reader.result as string);
                reader.readAsText(data);
            }
        };
        websocket.onclose = () => {
            options?.onClose?.();
            socket.value = undefined;
            status.value = "closed";
            errors.value = [];
        };
        websocket.onerror = error => {
            errors.value.push(error);
            options?.onError?.(error);
        };

        socket.value = websocket;
        status.value = "connecting";
    };
    const sendMessage = (message: string | ArrayBufferLike | Blob | ArrayBufferView) => {
        if (socket.value && socket.value.readyState === WebSocket.OPEN) {
            socket.value.send(message);
        }
    };

    return {
        url,
        status,
        errors,
        connect,
        sendMessage,
        disconnect,
        subscribe
    };
});

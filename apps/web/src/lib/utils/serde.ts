const dataTypes = {
    float32: Float32Array,
    uint8: Uint8Array,
    int8: Int8Array,
    uint16: Uint16Array,
    int16: Int16Array,
    int32: Int32Array,
    int64: BigInt64Array,
    bool: Uint8Array,
    float64: Float64Array,
    uint32: Uint32Array,
    uint64: BigUint64Array
};

export const arrayToB64 = (array: ArrayBufferView) => {
    // Use the view's own offset/length — TypedArrays returned from ORT may
    // share an underlying buffer with other data.
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    // `String.fromCharCode.apply(null, bytes)` passes every byte as a
    // separate argument; on large payloads (the 400 KB training batches
    // we ship over WS) that overflows V8's call stack. Chunk in
    // 32 KB blocks — well under the limit, still only a handful of
    // calls for typical inputs.
    const CHUNK = 0x8000;
    let binary = "";
    for (let i = 0; i < bytes.length; i += CHUNK) {
        binary += String.fromCharCode.apply(
            null,
            bytes.subarray(i, i + CHUNK) as unknown as number[]
        );
    }
    return btoa(binary);
};

export const b64ToArray = (
    str: string,
    constructor: { new (buffer: ArrayBuffer): ArrayBufferView } | keyof typeof dataTypes
) => {
    const binary = atob(str);
    const buffer = new ArrayBuffer(binary.length);
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    const ctor = typeof constructor === "string" ? dataTypes[constructor] : constructor;
    return new ctor(buffer);
};

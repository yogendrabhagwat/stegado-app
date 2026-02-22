/**
 * Steganography Logic Module (Preserved from original)
 * Implements Least Significant Bit (LSB) encoding/decoding (client-side demo).
 * The real AI+AES+ECC processing is done server-side.
 */
const Steganography = {
    encode: function (imageFile, message) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = img.width; canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const data = imageData.data;
                    const fullMessage = message + '\0';
                    const binaryMessage = this.textToBinary(fullMessage);
                    if (binaryMessage.length > data.length * 0.75) {
                        reject(new Error("Message is too long for this image.")); return;
                    }
                    let dataIndex = 0;
                    for (let i = 0; i < binaryMessage.length; i++) {
                        if ((dataIndex + 1) % 4 === 0) dataIndex++;
                        let val = data[dataIndex] & 254;
                        val = val | parseInt(binaryMessage[i]);
                        data[dataIndex] = val; dataIndex++;
                    }
                    ctx.putImageData(imageData, 0, 0);
                    resolve(canvas.toDataURL('image/png'));
                };
                img.onerror = reject; img.src = e.target.result;
            };
            reader.onerror = reject; reader.readAsDataURL(imageFile);
        });
    },
    decode: function (imageFile) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = img.width; canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const data = imageData.data;
                    let binaryMessage = '', currentByte = '', dataIndex = 0;
                    while (dataIndex < data.length) {
                        if ((dataIndex + 1) % 4 === 0) { dataIndex++; continue; }
                        const bit = data[dataIndex] & 1;
                        currentByte += bit;
                        if (currentByte.length === 8) {
                            const charCode = parseInt(currentByte, 2);
                            if (charCode === 0) break;
                            binaryMessage += String.fromCharCode(charCode);
                            currentByte = '';
                        }
                        dataIndex++;
                    }
                    resolve(binaryMessage);
                };
                img.onerror = reject; img.src = e.target.result;
            };
            reader.onerror = reject; reader.readAsDataURL(imageFile);
        });
    },
    textToBinary: function (text) {
        let binary = '';
        for (let i = 0; i < text.length; i++) {
            let bin = text.charCodeAt(i).toString(2);
            binary += "0".repeat(8 - bin.length) + bin;
        }
        return binary;
    }
};
window.Steganography = Steganography;

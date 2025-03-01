import java.io.*;
import javax.sound.sampled.*;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;

public class FFTTimes {

    public static void main(String[] args) throws IOException {
        String inputFilename = "input.wav";
        String hFilename = "filter.wav";
        String outputFilename = "output_filtered.wav";

        // read input files
        double[] signal = null;
        AudioFormat format = null;
        try {
            File inFile = new File(inputFilename);
            AudioInputStream ais = AudioSystem.getAudioInputStream(inFile);
            format = ais.getFormat();
            signal = readAudioData(ais, format);
            ais.close();
        } catch (Exception e) {
            System.err.println("Error reading input signal: " + e);
            return;
        }

        double[] h = null;
        try {
            File inFile = new File(hFilename);
            AudioInputStream ais = AudioSystem.getAudioInputStream(inFile);
            format = ais.getFormat();
            h = readAudioData(ais, format);
            ais.close();
        } catch (Exception e) {
            System.err.println("Error reading filter: " + e);
            return;
        }



        long start, finish, timeElapsed;
        double[] filteredSignal = new double[0];
        StringBuilder czasy = new StringBuilder();
        long absoluteStart = System.currentTimeMillis();

        for(int i=0; i<50; i++){
            start = System.currentTimeMillis();
            // overlap-add
            filteredSignal = fftfilt(h, signal);
            finish = System.currentTimeMillis();
            timeElapsed = finish - start;
            czasy.append((timeElapsed/1000.0) + "\n");
            if(System.currentTimeMillis()>absoluteStart+60000){
                break;
            }
        }
        FileWriter fw = new FileWriter("times" + h.length + ".txt");
        fw.write(czasy.toString());
        fw.close();
        System.out.println("skończono");
        try {
            writeAudioData(outputFilename, filteredSignal, format);
        } catch (Exception e) {
            System.err.println("Error saving to file: " + e);
        }

        System.out.println("Filtering successful.");
    }



    public static double[] fftfilt(double[] b, double[] x) {
        int nx = x.length;
        int nb = b.length;
        double[] y = new double[nx];

        if (nb >= nx) {
            int nfft = nextPow2(nx + nb - 1);
            return fftBlockConvolve(b, x, nfft, nb, nx, nx);
        } else {
            // L - block length - change as needed
            // int L = 1024;
            int L = b.length;
            int nfft = nextPow2(L + nb - 1);
            DoubleFFT_1D fft = new DoubleFFT_1D(nfft);

            // array H of format: [Re0, Im0, Re1, Im1, ...])
            double[] H = new double[2 * nfft];
            for (int i = 0; i < nb; i++) {
                H[2 * i] = b[i];
                H[2 * i + 1] = 0.0;
            }
            for (int i = nb; i < nfft; i++) {
                H[2 * i] = 0.0;
                H[2 * i + 1] = 0.0;
            }
            fft.complexForward(H);

            // Output buffer
            double[] output = new double[nx];
            int i = 0;
            while (i < nx) {
                int blockSize = Math.min(L, nx - i);
                // Complex output buffer made from zeros
                double[] X = new double[2 * nfft];
                for (int j = 0; j < blockSize; j++) {
                    X[2 * j] = x[i + j];
                    X[2 * j + 1] = 0.0;
                }
                fft.complexForward(X);

                // Complex multiplication: X = X .* H
                for (int k = 0; k < nfft; k++) {
                    double xr = X[2 * k];
                    double xi = X[2 * k + 1];
                    double hr = H[2 * k];
                    double hi = H[2 * k + 1];
                    double pr = xr * hr - xi * hi;
                    double pi = xr * hi + xi * hr;
                    X[2 * k] = pr;
                    X[2 * k + 1] = pi;
                }

                fft.complexInverse(X, true);
                int convLength = Math.min(blockSize + nb - 1, nfft);
                // Overlap-add
                for (int j = 0; j < convLength; j++) {
                    if (i + j < nx) {
                        output[i + j] += X[2 * j];  // real part only
                    }
                }
                i += L;
            }
            return output;
        }
    }

    private static double[] fftBlockConvolve(double[] b, double[] x, int nfft, int nb, int nx, int L) {
        DoubleFFT_1D fft = new DoubleFFT_1D(nfft);
        double[] H = new double[2 * nfft];
        for (int i = 0; i < nb; i++) {
            H[2 * i] = b[i];
            H[2 * i + 1] = 0.0;
        }
        for (int i = nb; i < nfft; i++) {
            H[2 * i] = 0.0;
            H[2 * i + 1] = 0.0;
        }
        fft.complexForward(H);

        double[] output = new double[nx];
        int i = 0;
        while (i < nx) {
            int blockSize = Math.min(L, nx - i);
            double[] X = new double[2 * nfft];
            for (int j = 0; j < blockSize; j++) {
                X[2 * j] = x[i + j];
                X[2 * j + 1] = 0.0;
            }
            fft.complexForward(X);
            for (int k = 0; k < nfft; k++) {
                double xr = X[2 * k];
                double xi = X[2 * k + 1];
                double hr = H[2 * k];
                double hi = H[2 * k + 1];
                double pr = xr * hr - xi * hi;
                double pi = xr * hi + xi * hr;
                X[2 * k] = pr;
                X[2 * k + 1] = pi;
            }
            fft.complexInverse(X, true);
            int convLength = Math.min(blockSize + nb - 1, nfft);
            for (int j = 0; j < convLength; j++) {
                if (i + j < nx) {
                    output[i + j] += X[2 * j];
                }
            }
            i += L;
        }
        return output;
    }

    /**
     * Returns lowest power of 2 greater or equal to n
     */
    private static int nextPow2(int n) {
        int p = 1;
        while (p < n) {
            p *= 2;
        }
        return p;
    }

    public static double[] readAudioData(AudioInputStream ais, AudioFormat format) throws IOException {
        int frameSize = format.getFrameSize();
        int numFrames = (int) ais.getFrameLength();
        byte[] audioBytes = new byte[numFrames * frameSize];

        int bytesRead = 0;
        int offset = 0;
        while (offset < audioBytes.length && (bytesRead = ais.read(audioBytes, offset, audioBytes.length - offset)) != -1) {
            offset += bytesRead;
        }

        // 16-bit PCM
        int numSamples = numFrames * format.getChannels();
        double[] samples = new double[numFrames];
        for (int i = 0; i < numFrames; i++) {
            int idx = i * frameSize;
            int lo = audioBytes[idx] & 0xff;
            int hi = audioBytes[idx + 1];
            int sample = (hi << 8) | lo;
            samples[i] = sample / 32768.0;
        }
        return samples;
    }

    public static void writeAudioData(String filename, double[] samples, AudioFormat format) throws IOException {
        int numFrames = samples.length;
        int frameSize = format.getFrameSize();
        byte[] audioBytes = new byte[numFrames * frameSize];

        for (int i = 0; i < numFrames; i++) {

            int s = (int) Math.round(samples[i] * 32767.0);
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            audioBytes[2 * i] = (byte) (s & 0xff);
            audioBytes[2 * i + 1] = (byte) ((s >> 8) & 0xff);
        }

        ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
        AudioInputStream outAIS = new AudioInputStream(bais, format, numFrames);
        File outFile = new File(filename);
        AudioSystem.write(outAIS, AudioFileFormat.Type.WAVE, outFile);
        outAIS.close();
    }
}

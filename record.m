clear all;
clc;

%realtek = 1, wo mic = 2
fs = 16000;
rec = audiorecorder(fs, 16, 1, 2);
disp('Start recording')
recordblocking(rec, 60);
disp('End recording')

y = getaudiodata(rec);

%audiowrite('C:\Users\User\Desktop\Speech_Recognition_MatLab\sounds\test.wav', y, 44100);
%[y,fs] = audioread('C:\Users\User\Desktop\Speech_Recognition_MatLab\sounds\test.wav');

%lefty = y(:,2);
%righty = y(:,1);

dt = 1/fs;
t = 0:dt:(length(y)*dt)-dt;
asec = length(t)/60;

t = 0:dt:(length(y)*dt)-dt;
plot(t,y);
xlabel('seconds');
ylabel('amplitude');
title('y');

index = 1;
filecount = 963; 
letter = 'Z';
tmpy = zeros(floor(asec),1);

window=hamming(512); %%window with size of 512 points
noverlap=256; %%the number of points for repeating the window
nfft=1024; %%size of the fit

while index <= length(y)
    if (y(floor(index)) >= 0.01) && (((index+asec)*dt) < 60) && floor(index-(asec/4)) > 300
        count = 1;
        tmp = floor(index-(asec/4));
        e = tmp + asec;
        
        while tmp <= e
            tmpy(count) = y(tmp);
            tmp = tmp + 1;
            count = count + 1; 
        end
        
        filename = "D:\Speech_Recognition_MatLab\strokes\"+letter+"\"+letter+filecount+".wav";
        audiowrite(filename,tmpy,fs)
        if (tmpy(1) > 0.01)
            figure
            title(letter+filecount);
            subplot(1,2,1)
            t = 0:dt:(length(tmpy)*dt)-dt;
            plot(t,tmpy)
            axis tight
            %plot spectrogram 
            subplot(1,2,2)
            [S,F,T,P] = spectrogram(tmpy,window,noverlap,nfft,fs,'yaxis');
            surf(T,F,10*log10(P),'edgecolor','none'); axis tight;view(0,90);
        end
        filecount = filecount + 1;
        index = index + asec;
    end
    
    index = index + 1;
end
filecount
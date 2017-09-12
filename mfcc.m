function [ feature ] = mfcc( f )
% Load a speech waveform
[d,sr] = audioread(f);

% Calculate 12th order PLP features without RASTA
ord = 12;
[cep, spec] = melfcc(d,sr,'preemph',0,'modelorder',ord,'numcep',ord+1,...
                'dcttype',1,'dither',1,'nbands',ceil(hz2bark(sr/2))+1,...
                'fbtype','bark','usecmp',1,'wintime',0.032,'hoptime',0.016);

del = deltas(cep);
% Double deltas are deltas applied twice with a shorter window
ddel = deltas(deltas(cep,5),5);
% Composite, 39-element feature vector, just like we use for speech recognition
feature = [cep;del;ddel];

end

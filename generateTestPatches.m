%sliding window on test image to generate small patches
row = 1152 ;col = 1500; %row = 500 ;col = 500;
wsize = 25; half = 12;
stepSize = 6;
Npat = floor(col/stepSize*row);
patches = zeros(Npat, wsize^2);
labels = zeros(Npat, 1);
xcords = zeros(Npat, 1);
ycords = zeros(Npat, 1);
threshold = 5;%20;



%addpath ../groundTruthXML/;

p = 1;
for imageN = 6:6%1:1   % only one image, image 1, saves I image
    COOR = load(sprintf('C:\\Users\\Mingtai\\Desktop\\neweyework\\groundtruth25confidence\\coordinate%d.txt',imageN));  % know groundtruth
    if imageN<10
        Ia = imread(sprintf('C:\\Users\\Mingtai\\Desktop\\neweyework\\ddb1_fundusimages\\image00%d.png',imageN));
    else
        Ia = imread(sprintf('C:\\Users\\Mingtai\\Desktop\\neweyework\\ddb1_fundusimages\\image0%d.png',imageN));
    end
    I = im2double(Ia(:,:,2));
    
    for x = half+1:row-half % row, dont start at the start of image and dont end at the end
        for y = half+1:stepSize:col-half % col , 6 is a big jump
            patchG = I(x-half:x+half,y-half:y+half);
            patchG = imadjust(patchG);%imadjust(patchG,[0.01 0.4],[]);
            patches(p,:) = reshape(patchG,1,wsize*wsize);
            xcords(p,1) = x;
            ycords(p,1) = y;
            if (size(COOR,1)>0)
                Dis =  abs([COOR(:,2)-x,COOR(:,1)-y]);
                minD = min(Dis(:,1)+Dis(:,2));
                if (minD < threshold)
                    labels(p,1) = 1; %MA
                else
                    labels(p,1) = 0; % NOT MA
                end
            else
                labels(p,1) = 0;
            end
            p=p+1;
        end
    end
    
end

% totalInstance = min(find(labels==0))-1;
% labels = labels(1:totalInstance,:);
% patches = patches(1:totalInstance,:);

save('test_86.mat','patches','labels','xcords','ycords');
csvwrite('datatest.csv',patches);
csvwrite('labelstest.csv',labels);
csvwrite('xmatlabcord.csv',xcords);
csvwrite('ymatlabcord.csv',ycords);


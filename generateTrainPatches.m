wsize = 25;
half = floor(wsize/2);
% addpath
trainingP = zeros(1,wsize*wsize);
addpath ../groundTruthXML/;

p = 1;
for imageN = 11:79%1:72

    COOR = load(sprintf('./groundtruth25confidence/coordinate%d.txt',imageN));
    if imageN<10
        I = imread(sprintf('./ddb1_fundusimages/image00%d.png',imageN));
    else
        I = imread(sprintf('./ddb1_fundusimages/image0%d.png',imageN));
    end
    I = im2double(I);

    row =size(I,1); col = size(I,2);
    numP = size(COOR,1);

    for i=1:numP
        y = COOR(i,1);
        x = COOR(i,2);

        [flag, swiftPatch]=outBounds(x,y, wsize, I);
        if flag==false
            patchG = I(x-half:x+half,y-half:y+half,2);
             patchG = imadjust(patchG);%imadjust(patchG,[0.01 0.4],[]);
            trainingP(p,:) = reshape(patchG,1,wsize*wsize);
        else
            swiftPatch = imadjust(swiftPatch);%,[0.01 0.4],[]);
            trainingP(p,:) = swiftPatch;
        end
        p=p+1;
    end
end
numberP = p-1;

data = trainingP;
%pause;
%% generate negative data



for imageN = 11:79%:72

    COOR = load(sprintf('./groundtruth25confidence/coordinate%d.txt',imageN));
    if imageN<10
        I = imread(sprintf('./ddb1_fundusimages/image00%d.png',imageN));
    else
        I = imread(sprintf('./ddb1_fundusimages/image0%d.png',imageN));
    end
    I = im2double(I);
    mask = imread(sprintf('./ddb1_fundusmask/fmask.tif'));

    row =size(I,1); col = size(I,2);
    numP = size(COOR,1);

    NCoor = [];
    for i=1:100
        minD = 0;
        while (minD < wsize )
            y = randi(col-wsize-40-50,1)+half+40; %exclude the black border on both sides
            x = randi(row-wsize,1)+half;

            while(mask(x,y)==0)
                y = randi(col-wsize-40-50,1)+half+40; %exclude the black border on both sides
                x = randi(row-wsize,1)+half;
            end
            if (size(COOR,1)>0)
                Dis =  abs([COOR(:,2)-x,COOR(:,1)-y]);
                minD = min(Dis(:,1)+Dis(:,2));
            else
                minD = wsize+1;
            end
        end

        NCoor = [NCoor;[x,y]];
        patchG = I(x-half:x+half,y-half:y+half,2);
         patchG = imadjust(patchG);
%         figure;imshow(patchG);
        data(p,:) = reshape(patchG,1,wsize*wsize);
        p=p+1;
    end

end

display(numberP);
postive = ones(numberP,1);

negative = zeros(p-1-numberP,1);

labels = [postive;negative];
save('trainMA25_140_Ori.mat','data','postive','negative','labels');
csvwrite('datatrain.csv',data);
csvwrite('labelstrain.csv',labels);

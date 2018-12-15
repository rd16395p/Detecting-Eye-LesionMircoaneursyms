 function [F,S] = outBounds(x,y,wsize,I)
 half = floor(wsize/2);
 row =size(I,1); col = size(I,2);
 F = false;
 if(x <= half)
     x = half+1;
     F = true;
 end
 if(y<=half)
     y = half+1;
     F = true;
 end
 if(x+half>row)
     x = row-half;
     F = true;
 end
 if(y+half>col)
     y = col - half;
     F = true;
 end
     
 patchG = I(x-half:x+half,y-half:y+half,2);
 S = reshape(patchG,1,wsize*wsize);
 end
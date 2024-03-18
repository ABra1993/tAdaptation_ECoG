%%% creates scrambled images from a set of intact images by swapping x by x
%%% pixel 'blocks' across images 
%%% position is kept constant 

%%% IG NIMH 2015

%%%% set parameters here %%%%
nNew = 240; % number of new images wanted; can go up to 1000
imsize = 768; % size of input images (now square only
blocksize = 48; % size of blocks (now square only)

% original image directory:
imDir = '/Users/groenii/Experiments_local/6categorylocalizer/stims/objects';
% directory in which to place the scrambled images
newImDir = '/Users/groenii/Experiments_local/6categorylocalizer/stims/scrambled';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd(imDir)

% get a list of all the images in the imDir
D = dir('*.jpg');
nIms = length(D);

A = nan(imsize,imsize,3,nIms);
% read all images into one big array A
for cIm = 1:nIms
    Im = imread(D(cIm).name);
    if size(Im,3) ~= 3
        for cLayer = 1:3
            A(:,:,cLayer,cIm) = Im;
        end
    else
        A(:,:,:,cIm) = Im; 
    end
end

% create empty image to fill with blocks
newIm = nan(imsize,imsize,3);
blockind = 1:blocksize:imsize+blocksize;
cd(newImDir);
rng('shuffle'); % reset random generator

% for each block in each image, grab random block
% (RGB layers are maintained; images are grabbed with replacement)
for cIm = 1:nNew
    for cBlock1 = 1:length(blockind)-1
        for cBlock2 = 1:length(blockind)-1
            xblock = blockind(cBlock1):blockind(cBlock1+1)-1;
            yblock = blockind(cBlock2):blockind(cBlock2+1)-1;
            randorder = randperm(nIms);
            newIm(xblock,yblock,:) = squeeze(A(xblock,yblock,:,randorder(1)));            
        end
    end
    % display new image
    %figure;image(uint8(newIm));
    % generate name for new image
    if cIm < 10
        newName = ['scrambled00' num2str(cIm) '.jpg'];
    elseif cIm < 100
        newName = ['scrambled0' num2str(cIm) '.jpg'];
    else
        newName = ['scrambled' num2str(cIm) '.jpg'];
    end
    % write image to file
    imwrite(uint8(newIm), newName, 'jpg');
end

    
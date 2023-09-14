docker run -i -d --rm --runtime=nvidia --name dac \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
                    -v ~/:/root/ \
                    gitlab-registry.nrp-nautilus.io/openhonor/diffusion-policies-for-offline-rl
           
docker exec -it dac bash
# -e QT_X11_NO_MITSHM=1 -e XAUTHORITY 

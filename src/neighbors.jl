"""
    Neighbor

Represents a communications neighbor. Constructors are defined by the
backends and require a unique ID that represents the neighbor, the array
type to be used for staging (usually `Array` on the CPU and `CuArray`
for GPU code), the element type for the array, and the dimensions for
the send and receive buffers.
"""
struct Neighbor{B}
    send_buf::B
    recv_buf::B

    function Neighbor{B}(send_buf, recv_buf) where {B}
        new{B}(send_buf, recv_buf)
    end
end
Neighbor(::Type{CC}, x...) where {CC <: AbstractCommsContext} =
    error("No `Neighbor` constructor defined for $CC")
send_buffer(nbr::Neighbor) = nbr.send_buf
recv_buffer(nbr::Neighbor) = nbr.recv_buf

"""
    send_stage(neighbor::Neighbor)

Return the staging area for data that is to be sent to the specified
neighbor.
"""
send_stage(neighbor::Neighbor) = get_stage(neighbor.send_buf)

"""
    recv_stage(neighbor::Neighbor)

Return the staging area for data that has been received from the
specified neighbor.
"""
recv_stage(neighbor::Neighbor) = get_stage(neighbor.recv_buf)

package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestHexKey(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(1, HexKey("AB"))
    assert.Equal(2, HexKey("1077E"))
    assert.Equal(4, HexKey("ABED1A33"))
    assert.Equal(2, HexKey("2020"))
    assert.Equal(6, HexKey("123456789ABCDEF0"))
    assert.Equal(12, HexKey("112233445566778899AABBCCDDEEFF00"))
    assert.Equal(0, HexKey(""))
}

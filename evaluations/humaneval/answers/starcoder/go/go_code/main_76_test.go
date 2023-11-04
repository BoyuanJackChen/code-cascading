package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestIsSimplePower(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, IsSimplePower(1, 4))
    assert.Equal(true, IsSimplePower(2, 2))
    assert.Equal(true, IsSimplePower(8, 2))
    assert.Equal(false, IsSimplePower(3, 2))
    assert.Equal(false, IsSimplePower(3, 1))
    assert.Equal(false, IsSimplePower(5, 3))
    assert.Equal(true, IsSimplePower(16, 2))
    assert.Equal(false, IsSimplePower(143214, 16))
    assert.Equal(true, IsSimplePower(4, 2))
    assert.Equal(true, IsSimplePower(9, 3))
    assert.Equal(true, IsSimplePower(16, 4))
    assert.Equal(false, IsSimplePower(24, 2))
    assert.Equal(false, IsSimplePower(128, 4))
    assert.Equal(false, IsSimplePower(12, 6))
    assert.Equal(true, IsSimplePower(1, 1))
    assert.Equal(true, IsSimplePower(1, 12))
}

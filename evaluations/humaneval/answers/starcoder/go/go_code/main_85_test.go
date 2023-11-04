package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestAdd(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(88, Add([]int{4, 88}))
    assert.Equal(122, Add([]int{4, 5, 6, 7, 2, 122}))
    assert.Equal(0, Add([]int{4, 0, 6, 7}))
    assert.Equal(12, Add([]int{4, 4, 6, 8}))
}
    
